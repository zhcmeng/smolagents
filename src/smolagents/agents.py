#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
智能代理模块 - smolagents 的核心代理实现

本模块包含了基于 ReAct 框架的多步骤智能代理实现，支持工具调用和代码执行两种模式。
主要类：
- MultiStepAgent: 多步骤代理的抽象基类
- ToolCallingAgent: 基于工具调用的代理
- CodeAgent: 基于代码执行的代理

作者: HuggingFace 团队
版本: 1.0
"""

import importlib
import inspect
import json
import os
import re
import tempfile
import textwrap
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import jinja2
import yaml
from huggingface_hub import create_repo, metadata_update, snapshot_download, upload_folder
from jinja2 import StrictUndefined, Template
from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text


if TYPE_CHECKING:
    import PIL.Image

from .agent_types import AgentAudio, AgentImage, handle_agent_output_types
from .default_tools import TOOL_MAPPING, FinalAnswerTool
from .local_python_executor import BASE_BUILTIN_MODULES, LocalPythonExecutor, PythonExecutor, fix_final_answer_code
from .memory import (
    ActionStep,
    AgentMemory,
    FinalAnswerStep,
    Message,
    PlanningStep,
    SystemPromptStep,
    TaskStep,
    Timing,
    TokenUsage,
    ToolCall,
)
from .models import (
    CODEAGENT_RESPONSE_FORMAT,
    ChatMessage,
    ChatMessageStreamDelta,
    MessageRole,
    Model,
    parse_json_if_needed,
)
from .monitoring import (
    YELLOW_HEX,
    AgentLogger,
    LogLevel,
    Monitor,
)
from .remote_executors import DockerExecutor, E2BExecutor
from .tools import Tool
from .utils import (
    AgentError,
    AgentExecutionError,
    AgentGenerationError,
    AgentMaxStepsError,
    AgentParsingError,
    AgentToolCallError,
    AgentToolExecutionError,
    extract_code_from_text,
    is_valid_name,
    make_init_file,
    parse_code_blobs,
    truncate_content,
)


logger = getLogger(__name__)


def get_variable_names(self, template: str) -> set[str]:
    """
    从模板字符串中提取变量名称
    
    参数:
        template (str): Jinja2 模板字符串
        
    返回:
        set[str]: 模板中使用的变量名称集合
    """
    pattern = re.compile(r"\{\{([^{}]+)\}\}")
    return {match.group(1).strip() for match in pattern.finditer(template)}


def populate_template(template: str, variables: dict[str, Any]) -> str:
    """
    使用提供的变量填充 Jinja2 模板
    
    参数:
        template (str): Jinja2 模板字符串
        variables (dict[str, Any]): 用于填充模板的变量字典
        
    返回:
        str: 填充后的字符串
        
    异常:
        Exception: 当模板渲染失败时抛出异常
    """
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")


@dataclass
class FinalOutput:
    """
    代理执行的最终输出结果
    
    属性:
        output (Any | None): 代理执行的最终输出，可以是任何类型或 None
    """
    output: Any | None


class PlanningPromptTemplate(TypedDict):
    """
    Prompt templates for the planning step.

    Args:
        plan (`str`): Initial plan prompt.
        update_plan_pre_messages (`str`): Update plan pre-messages prompt.
        update_plan_post_messages (`str`): Update plan post-messages prompt.
    """

    initial_plan: str
    update_plan_pre_messages: str
    update_plan_post_messages: str


class ManagedAgentPromptTemplate(TypedDict):
    """
    Prompt templates for the managed agent.

    Args:
        task (`str`): Task prompt.
        report (`str`): Report prompt.
    """

    task: str
    report: str


class FinalAnswerPromptTemplate(TypedDict):
    """
    Prompt templates for the final answer.

    Args:
        pre_messages (`str`): Pre-messages prompt.
        post_messages (`str`): Post-messages prompt.
    """

    pre_messages: str
    post_messages: str


class PromptTemplates(TypedDict):
    """
    Prompt templates for the agent.

    Args:
        system_prompt (`str`): System prompt.
        planning ([`~agents.PlanningPromptTemplate`]): Planning prompt templates.
        managed_agent ([`~agents.ManagedAgentPromptTemplate`]): Managed agent prompt templates.
        final_answer ([`~agents.FinalAnswerPromptTemplate`]): Final answer prompt templates.
    """

    system_prompt: str
    planning: PlanningPromptTemplate
    managed_agent: ManagedAgentPromptTemplate
    final_answer: FinalAnswerPromptTemplate


EMPTY_PROMPT_TEMPLATES = PromptTemplates(
    system_prompt="",
    planning=PlanningPromptTemplate(
        initial_plan="",
        update_plan_pre_messages="",
        update_plan_post_messages="",
    ),
    managed_agent=ManagedAgentPromptTemplate(task="", report=""),
    final_answer=FinalAnswerPromptTemplate(pre_messages="", post_messages=""),
)


@dataclass
class RunResult:
    """Holds extended information about an agent run.

    Attributes:
        output (Any | None): The final output of the agent run, if available.
        state (Literal["success", "max_steps_error"]): The final state of the agent after the run.
        messages (list[dict]): The agent's memory, as a list of messages.
        token_usage (TokenUsage | None): Count of tokens used during the run.
        timing (Timing): Timing details of the agent run: start time, end time, duration.
    """

    output: Any | None
    state: Literal["success", "max_steps_error"]
    messages: list[dict]
    token_usage: TokenUsage | None
    timing: Timing


class MultiStepAgent(ABC):
    """
    多步骤智能代理抽象基类 - 基于 ReAct 框架的任务解决器
    
    该类使用 ReAct（推理-行动）框架逐步解决给定任务：
    在目标未达成时，代理将执行由 LLM 生成的行动和从环境获得的观察的循环。
    
    主要特性:
    - 支持多种工具调用
    - 支持子代理管理
    - 支持步骤回调和规划间隔
    - 支持最终答案验证
    - 支持流式输出
    
    参数:
        tools (`list[Tool]`): 代理可以使用的工具列表
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): 生成代理行动的模型
        prompt_templates ([`~agents.PromptTemplates`], *可选*): 提示模板集合
        max_steps (`int`, 默认 `20`): 代理解决任务的最大步数
        add_base_tools (`bool`, 默认 `False`): 是否添加基础工具到代理的工具集
        verbosity_level (`LogLevel`, 默认 `LogLevel.INFO`): 代理日志的详细程度
        grammar (`dict[str, str]`, *可选*): 用于解析 LLM 输出的语法规则
            <已弃用 版本="1.17.0">
            参数 `grammar` 已弃用，将在版本 1.20 中移除。
            </已弃用>
        managed_agents (`list`, *可选*): 代理可以调用的子代理列表
        step_callbacks (`list[Callable]`, *可选*): 每步执行时调用的回调函数列表
        planning_interval (`int`, *可选*): 执行规划步骤的间隔
        name (`str`, *可选*): 子代理必需 - 该代理被调用时的名称
        description (`str`, *可选*): 子代理必需 - 该代理的描述
        provide_run_summary (`bool`, *可选*): 作为子代理被调用时是否提供运行摘要
        final_answer_checks (`list[Callable]`, *可选*): 接受最终答案前运行的验证函数列表
            每个函数应该:
            - 接受最终答案和代理的记忆作为参数
            - 返回一个布尔值，指示最终答案是否有效
    """

    def __init__(
        self,
        tools: list[Tool],
        model: Model,
        prompt_templates: PromptTemplates | None = None,
        max_steps: int = 20,
        add_base_tools: bool = False,
        verbosity_level: LogLevel = LogLevel.INFO,
        grammar: dict[str, str] | None = None,
        managed_agents: list | None = None,
        step_callbacks: list[Callable] | None = None,
        planning_interval: int | None = None,
        name: str | None = None,
        description: str | None = None,
        provide_run_summary: bool = False,
        final_answer_checks: list[Callable] | None = None,
        return_full_result: bool = False,
        logger: AgentLogger | None = None,
    ):
        """
        初始化多步骤智能代理
        
        该构造函数负责初始化代理的所有核心组件，包括工具、模型、提示模板、
        管理代理、回调函数等。同时进行必要的验证和配置检查。
        
        初始化流程:
        1. 设置基本属性（模型、提示模板、最大步数等）
        2. 验证并设置提示模板
        3. 配置管理代理和工具
        4. 验证配置的一致性
        5. 初始化内存和日志系统
        6. 设置监控和回调机制
        
        参数:
            tools (list[Tool]): 代理可以使用的工具列表
            model (Model): 用于生成代理响应的语言模型
            prompt_templates (PromptTemplates | None): 提示模板配置
            max_steps (int): 解决任务的最大步数限制
            add_base_tools (bool): 是否添加基础工具集
            verbosity_level (LogLevel): 日志详细程度级别
            grammar (dict[str, str] | None): 已弃用的语法规则配置
            managed_agents (list | None): 子代理列表
            step_callbacks (list[Callable] | None): 步骤执行回调函数列表
            planning_interval (int | None): 规划步骤执行间隔
            name (str | None): 代理名称（子代理必需）
            description (str | None): 代理描述（子代理必需）
            provide_run_summary (bool): 是否提供运行摘要
            final_answer_checks (list[Callable] | None): 最终答案验证函数列表
            return_full_result (bool): 是否返回完整结果对象
            logger (AgentLogger | None): 自定义日志记录器
        """
        # 设置基本代理属性
        self.agent_name = self.__class__.__name__  # 代理类名称
        self.model = model  # 语言模型实例
        
        # 配置提示模板，使用默认模板或用户提供的模板
        self.prompt_templates = prompt_templates or EMPTY_PROMPT_TEMPLATES
        
        # 验证用户提供的提示模板完整性
        if prompt_templates is not None:
            # 检查是否缺少必需的顶级键
            missing_keys = set(EMPTY_PROMPT_TEMPLATES.keys()) - set(prompt_templates.keys())
            assert not missing_keys, (
                f"Some prompt templates are missing from your custom `prompt_templates`: {missing_keys}"
            )
            
            # 验证嵌套字典结构的完整性
            for key, value in EMPTY_PROMPT_TEMPLATES.items():
                if isinstance(value, dict):
                    for subkey in value.keys():
                        assert key in prompt_templates.keys() and (subkey in prompt_templates[key].keys()), (
                            f"Some prompt templates are missing from your custom `prompt_templates`: {subkey} under {key}"
                        )

        # 设置执行控制参数
        self.max_steps = max_steps  # 最大执行步数
        self.step_number = 0  # 当前步数计数器
        
        # 处理已弃用的语法参数
        if grammar is not None:
            warnings.warn(
                "Parameter 'grammar' is deprecated and will be removed in version 1.20.",
                FutureWarning,
            )
        self.grammar = grammar  # 语法规则（已弃用）
        
        # 设置规划和状态管理
        self.planning_interval = planning_interval  # 规划执行间隔
        self.state: dict[str, Any] = {}  # 代理状态字典，存储执行过程中的变量
        
        # 设置代理身份信息
        self.name = self._validate_name(name)  # 验证并设置代理名称
        self.description = description  # 代理描述
        self.provide_run_summary = provide_run_summary  # 是否提供运行摘要
        self.final_answer_checks = final_answer_checks  # 最终答案验证函数列表
        self.return_full_result = return_full_result  # 是否返回详细结果

        # 初始化管理代理和工具
        self._setup_managed_agents(managed_agents)  # 设置子代理
        self._setup_tools(tools, add_base_tools)  # 设置工具集
        self._validate_tools_and_managed_agents(tools, managed_agents)  # 验证配置一致性

        # 初始化任务和内存系统
        self.task: str | None = None  # 当前执行的任务
        self.memory = AgentMemory(self.system_prompt)  # 代理记忆系统

        # 设置日志记录器
        if logger is None:
            # 使用默认日志记录器
            self.logger = AgentLogger(level=verbosity_level)
        else:
            # 使用用户提供的日志记录器
            self.logger = logger

        # 初始化监控和回调系统
        self.monitor = Monitor(self.model, self.logger)  # 性能监控器
        self.step_callbacks = step_callbacks if step_callbacks is not None else []  # 步骤回调函数列表
        self.step_callbacks.append(self.monitor.update_metrics)  # 添加监控更新回调
        self.stream_outputs = False  # 流式输出标志

    @property
    def system_prompt(self) -> str:
        """
        获取系统提示词
        
        该属性通过调用子类实现的 initialize_system_prompt 方法来生成
        完整的系统提示词。系统提示词定义了代理的身份、能力和行为规范。
        
        返回:
            str: 格式化后的系统提示词字符串
        """
        return self.initialize_system_prompt()

    @system_prompt.setter
    def system_prompt(self, value: str):
        """
        系统提示词设置器 - 只读属性
        
        系统提示词是只读属性，不能直接设置。如需修改系统提示词，
        应该修改 self.prompt_templates["system_prompt"] 的内容。
        
        参数:
            value (str): 尝试设置的值
            
        异常:
            AttributeError: 总是抛出，因为该属性是只读的
        """
        raise AttributeError(
            """The 'system_prompt' property is read-only. Use 'self.prompt_templates["system_prompt"]' instead."""
        )

    def _validate_name(self, name: str | None) -> str | None:
        """
        验证代理名称的有效性
        
        检查代理名称是否符合 Python 标识符规范且不是保留关键字。
        这对于子代理特别重要，因为它们的名称会用作函数调用的标识符。
        
        参数:
            name (str | None): 待验证的代理名称
            
        返回:
            str | None: 验证通过的名称或 None
            
        异常:
            ValueError: 当名称不符合 Python 标识符规范时抛出
        """
        if name is not None and not is_valid_name(name):
            raise ValueError(f"Agent name '{name}' must be a valid Python identifier and not a reserved keyword.")
        return name

    def _setup_managed_agents(self, managed_agents: list | None = None) -> None:
        """
        设置管理的子代理
        
        将子代理列表转换为字典形式以便快速查找，并验证每个子代理
        都具有必需的名称和描述信息。子代理可以作为工具被主代理调用。
        
        参数:
            managed_agents (list | None): 子代理列表
            
        异常:
            AssertionError: 当子代理缺少名称或描述时抛出
        """
        # 初始化空的管理代理字典
        self.managed_agents = {}
        
        if managed_agents:
            # 验证所有子代理都有名称和描述
            assert all(agent.name and agent.description for agent in managed_agents), (
                "All managed agents need both a name and a description!"
            )
            # 将子代理列表转换为以名称为键的字典
            self.managed_agents = {agent.name: agent for agent in managed_agents}

    def _setup_tools(self, tools, add_base_tools):
        """
        设置代理可用的工具集
        
        将工具列表转换为字典形式，可选择性添加基础工具集，
        并确保始终包含最终答案工具。
        
        参数:
            tools (list[Tool]): 用户提供的工具列表
            add_base_tools (bool): 是否添加基础工具集
            
        异常:
            AssertionError: 当工具列表中包含非 Tool 实例时抛出
        """
        # 验证所有元素都是 Tool 实例
        assert all(isinstance(tool, Tool) for tool in tools), "All elements must be instance of Tool (or a subclass)"
        
        # 将工具列表转换为以名称为键的字典
        self.tools = {tool.name: tool for tool in tools}
        
        # 根据需要添加基础工具集
        if add_base_tools:
            self.tools.update(
                {
                    name: cls()
                    for name, cls in TOOL_MAPPING.items()
                    # python_interpreter 只在 ToolCallingAgent 中添加
                    if name != "python_interpreter" or self.__class__.__name__ == "ToolCallingAgent"
                }
            )
        
        # 确保始终包含最终答案工具
        self.tools.setdefault("final_answer", FinalAnswerTool())

    def _validate_tools_and_managed_agents(self, tools, managed_agents):
        """
        验证工具和管理代理名称的唯一性
        
        确保所有工具、管理代理和当前代理的名称都是唯一的，
        避免在调用时产生歧义。
        
        参数:
            tools (list[Tool]): 工具列表
            managed_agents (list | None): 管理代理列表
            
        异常:
            ValueError: 当发现重复名称时抛出
        """
        # 收集所有工具名称
        tool_and_managed_agent_names = [tool.name for tool in tools]
        
        # 添加管理代理名称
        if managed_agents is not None:
            tool_and_managed_agent_names += [agent.name for agent in managed_agents]
        
        # 添加当前代理名称（如果存在）
        if self.name:
            tool_and_managed_agent_names.append(self.name)
        
        # 检查名称唯一性
        if len(tool_and_managed_agent_names) != len(set(tool_and_managed_agent_names)):
            raise ValueError(
                "Each tool or managed_agent should have a unique name! You passed these duplicate names: "
                f"{[name for name in tool_and_managed_agent_names if tool_and_managed_agent_names.count(name) > 1]}"
            )

    def run(
        self,
        task: str,
        stream: bool = False,
        reset: bool = True,
        images: list["PIL.Image.Image"] | None = None,
        additional_args: dict | None = None,
        max_steps: int | None = None,
    ):
        """
        为给定任务运行代理
        
        这是代理的主要执行方法，会启动完整的任务解决流程。
        
        参数:
            task (`str`): 要执行的任务描述
            stream (`bool`): 是否以流式模式运行
                如果为 `True`，返回一个生成器，逐步产出每个执行步骤。
                必须遍历此生成器来处理各个步骤（例如使用 for 循环或 `next()`）。
                如果为 `False`，内部执行所有步骤，完成后仅返回最终答案。
            reset (`bool`): 是否重置对话或从上次运行继续
            images (`list[PIL.Image.Image]`, *可选*): 图像对象列表
            additional_args (`dict`, *可选*): 要传递给代理运行的其他变量，
                例如图像或数据框。请为它们提供清晰的名称！
            max_steps (`int`, *可选*): 代理解决任务的最大步数。
                如果未提供，将使用代理的默认值。
        
        返回:
            如果 stream=False：返回最终答案
            如果 stream=True：返回步骤生成器
            如果 return_full_result=True：返回 RunResult 对象
        
        示例:
        ```python
        from smolagents import CodeAgent
        agent = CodeAgent(tools=[])
        result = agent.run("2 的 3.7384 次方是多少？")
        ```
        """
        max_steps = max_steps or self.max_steps
        self.task = task
        self.interrupt_switch = False
        if additional_args is not None:
            self.state.update(additional_args)
            self.task += f"""
You have been provided with these additional arguments, that you can access using the keys as variables in your python code:
{str(additional_args)}."""

        self.memory.system_prompt = SystemPromptStep(system_prompt=self.system_prompt)
        if reset:
            self.memory.reset()
            self.monitor.reset()

        self.logger.log_task(
            content=self.task.strip(),
            subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
            level=LogLevel.INFO,
            title=self.name if hasattr(self, "name") else None,
        )
        self.memory.steps.append(TaskStep(task=self.task, task_images=images))

        if getattr(self, "python_executor", None):
            self.python_executor.send_variables(variables=self.state)
            self.python_executor.send_tools({**self.tools, **self.managed_agents})

        if stream:
            # The steps are returned as they are executed through a generator to iterate on.
            return self._run_stream(task=self.task, max_steps=max_steps, images=images)
        run_start_time = time.time()
        # Outputs are returned only at the end. We only look at the last step.

        steps = list(self._run_stream(task=self.task, max_steps=max_steps, images=images))
        assert isinstance(steps[-1], FinalAnswerStep)
        output = steps[-1].output

        if self.return_full_result:
            total_input_tokens = 0
            total_output_tokens = 0
            correct_token_usage = True
            for step in self.memory.steps:
                if isinstance(step, (ActionStep, PlanningStep)):
                    if step.token_usage is None:
                        correct_token_usage = False
                        break
                    else:
                        total_input_tokens += step.token_usage.input_tokens
                        total_output_tokens += step.token_usage.output_tokens
            if correct_token_usage:
                token_usage = TokenUsage(input_tokens=total_input_tokens, output_tokens=total_output_tokens)
            else:
                token_usage = None

            if self.memory.steps and isinstance(getattr(self.memory.steps[-1], "error", None), AgentMaxStepsError):
                state = "max_steps_error"
            else:
                state = "success"

            messages = self.memory.get_full_steps()

            return RunResult(
                output=output,
                token_usage=token_usage,
                messages=messages,
                timing=Timing(start_time=run_start_time, end_time=time.time()),
                state=state,
            )

        return output

    def _run_stream(
        self, task: str, max_steps: int, images: list["PIL.Image.Image"] | None = None
    ) -> Generator[ActionStep | PlanningStep | FinalAnswerStep | ChatMessageStreamDelta]:
        """
        以流式方式运行代理的核心执行循环
        
        该方法实现了 ReAct 框架的完整执行流程，逐步生成并产出每个执行步骤。
        代理将持续执行思考-行动-观察的循环，直到获得最终答案或达到最大步数。
        
        执行流程:
        1. 初始化执行状态和步数计数器
        2. 主执行循环：
           - 检查中断信号
           - 根据规划间隔执行规划步骤
           - 执行行动步骤（思考、行动、观察）
           - 处理异常情况
           - 更新内存和步数
        3. 处理达到最大步数的情况
        4. 产出最终答案
        
        参数:
            task (str): 要执行的任务描述
            max_steps (int): 允许的最大执行步数
            images (list["PIL.Image.Image"] | None): 可选的图像输入列表
            
        产出:
            Generator: 逐步产出以下类型的对象：
                - ActionStep: 行动步骤，包含模型输出、工具调用、观察结果等
                - PlanningStep: 规划步骤，包含代理的规划思路和策略
                - FinalAnswerStep: 最终答案步骤，包含任务的最终结果
                - ChatMessageStreamDelta: 流式消息增量（如果启用流式输出）
        
        异常处理:
            - AgentGenerationError: 代理生成错误，通常由实现问题引起，直接抛出
            - AgentError: 其他代理错误，通常由模型引起，记录后继续执行
            - 其他异常: 在 finally 块中进行清理
        """
        # 初始化执行状态
        final_answer = None  # 最终答案，初始为 None
        self.step_number = 1  # 当前步数，从 1 开始
        
        # 主执行循环：持续执行直到获得最终答案或达到最大步数
        while final_answer is None and self.step_number <= max_steps:
            # 检查中断信号：如果用户请求中断，则立即停止执行
            if self.interrupt_switch:
                raise AgentError("Agent interrupted.", self.logger)

            # 规划步骤执行逻辑：根据设定的规划间隔执行规划
            # 规划步骤在第一步或每隔 planning_interval 步执行一次
            if self.planning_interval is not None and (
                self.step_number == 1 or (self.step_number - 1) % self.planning_interval == 0
            ):
                # 记录规划步骤的开始时间
                planning_start_time = time.time()
                planning_step = None
                
                # 生成并产出规划步骤的流式输出
                # _generate_planning_step 会产出流式消息和最终的规划步骤
                for element in self._generate_planning_step(
                    task, is_first_step=(self.step_number == 1), step=self.step_number
                ):
                    yield element  # 向外产出流式元素（如 ChatMessageStreamDelta）
                    planning_step = element  # 保存最后一个元素（应该是 PlanningStep）
                
                # 确保最后产出的元素确实是规划步骤
                assert isinstance(planning_step, PlanningStep)  # Last yielded element should be a PlanningStep
                
                # 将规划步骤添加到记忆中并记录执行时间
                self.memory.steps.append(planning_step)
                planning_end_time = time.time()
                planning_step.timing = Timing(
                    start_time=planning_start_time,
                    end_time=planning_end_time,
                )

            # 开始执行行动步骤
            action_step_start_time = time.time()
            
            # 创建行动步骤对象，包含步数、时间和观察图像
            action_step = ActionStep(
                step_number=self.step_number,
                timing=Timing(start_time=action_step_start_time),
                observations_images=images,
            )
            
            # 执行行动步骤的主要逻辑，包含异常处理
            try:
                # 执行单个行动步骤，可能产出流式消息和最终结果
                for el in self._execute_step(action_step):
                    yield el  # 向外产出流式元素
                final_answer = el  # 保存最后一个元素，可能是最终答案
                
            except AgentGenerationError as e:
                # 代理生成错误：这类错误通常由实现问题引起，不是模型错误
                # 需要直接抛出并退出，因为这表示程序逻辑有问题
                raise e
                
            except AgentError as e:
                # 其他代理错误：这类错误通常由模型引起（如解析错误、工具调用错误等）
                # 将错误记录到行动步骤中，但继续执行循环尝试恢复
                action_step.error = e
                
            finally:
                # 无论是否发生异常，都要执行清理工作
                # 完成行动步骤的收尾工作（设置结束时间、执行回调等）
                self._finalize_step(action_step)
                # 将行动步骤添加到记忆中
                self.memory.steps.append(action_step)
                # 向外产出完整的行动步骤对象
                yield action_step
                # 递增步数计数器，为下一轮循环做准备
                self.step_number += 1

        # 处理达到最大步数但仍未获得最终答案的情况
        if final_answer is None and self.step_number == max_steps + 1:
            # 强制生成最终答案
            final_answer = self._handle_max_steps_reached(task, images)
            yield action_step
            
        # 产出最终答案步骤
        # handle_agent_output_types 处理不同类型的输出（如图像、音频等）
        yield FinalAnswerStep(handle_agent_output_types(final_answer))

    def _execute_step(self, memory_step: ActionStep) -> Generator[ChatMessageStreamDelta | FinalOutput]:
        """
        执行单个行动步骤
        
        该方法负责执行代理的一个完整行动步骤，包括记录步骤信息、
        调用子类的步骤实现、处理流式输出和验证最终答案。
        
        执行流程:
        1. 记录步骤开始信息
        2. 调用子类的 _step_stream 方法执行具体逻辑
        3. 处理流式输出和最终答案
        4. 验证最终答案（如果存在验证函数）
        
        参数:
            memory_step (ActionStep): 当前执行的行动步骤对象

        产出:
            Generator[ChatMessageStreamDelta | FinalOutput]: 流式消息或最终输出
        """
        # 记录步骤开始
        self.logger.log_rule(f"Step {self.step_number}", level=LogLevel.INFO)
        
        # 执行步骤并处理输出
        for el in self._step_stream(memory_step):
            final_answer = el
            if isinstance(el, ChatMessageStreamDelta):
                # 流式消息直接向外传递
                yield el
            elif isinstance(el, FinalOutput):
                # 处理最终输出
                final_answer = el.output
                # 如果配置了最终答案验证函数，则进行验证
                if self.final_answer_checks:
                    self._validate_final_answer(final_answer)
                yield final_answer

    def _validate_final_answer(self, final_answer: Any):
        """
        验证最终答案的有效性
        
        依次调用所有配置的验证函数来检查最终答案是否符合要求。
        验证函数应该接受最终答案和代理内存作为参数，返回布尔值。
        
        参数:
            final_answer (Any): 待验证的最终答案
            
        异常:
            AgentError: 当任何验证函数失败时抛出
        """
        for check_function in self.final_answer_checks:
            try:
                # 调用验证函数，传入最终答案和代理内存
                assert check_function(final_answer, self.memory)
            except Exception as e:
                raise AgentError(f"Check {check_function.__name__} failed with error: {e}", self.logger)

    def _finalize_step(self, memory_step: ActionStep):
        """
        完成步骤的收尾工作
        
        设置步骤结束时间并执行所有配置的回调函数。
        回调函数用于监控、日志记录、指标更新等用途。
        
        参数:
            memory_step (ActionStep): 要完成的行动步骤对象
        """
        # 设置步骤结束时间
        memory_step.timing.end_time = time.time()
        
        # 执行所有步骤回调函数
        for callback in self.step_callbacks:
            # 兼容旧版本回调函数（只接受一个参数）和新版本（接受两个参数）
            callback(memory_step) if len(inspect.signature(callback).parameters) == 1 else callback(
                memory_step, agent=self
            )

    def _handle_max_steps_reached(self, task: str, images: list["PIL.Image.Image"]) -> Any:
        """
        处理达到最大步数的情况
        
        当代理达到最大步数限制但仍未获得最终答案时，强制生成
        一个最终答案。这确保代理总是能提供某种形式的回应。
        
        处理流程:
        1. 记录开始时间
        2. 调用 provide_final_answer 强制生成最终答案
        3. 创建带有错误标记的行动步骤
        4. 完成步骤并添加到内存
        5. 返回最终答案内容
        
        参数:
            task (str): 原始任务描述
            images (list["PIL.Image.Image"]): 任务相关的图像
            
        返回:
            Any: 强制生成的最终答案内容
        """
        # 记录最终答案生成的开始时间
        action_step_start_time = time.time()
        
        # 强制生成最终答案
        final_answer = self.provide_final_answer(task, images)
        
        # 创建标记为达到最大步数错误的行动步骤
        final_memory_step = ActionStep(
            step_number=self.step_number,
            error=AgentMaxStepsError("Reached max steps.", self.logger),
            timing=Timing(start_time=action_step_start_time, end_time=time.time()),
            token_usage=final_answer.token_usage,
        )
        
        # 设置行动输出为最终答案内容
        final_memory_step.action_output = final_answer.content
        
        # 完成步骤收尾工作并添加到内存
        self._finalize_step(final_memory_step)
        self.memory.steps.append(final_memory_step)
        
        return final_answer.content

    def _generate_planning_step(
        self, task, is_first_step: bool, step: int
    ) -> Generator[ChatMessageStreamDelta | PlanningStep]:
        start_time = time.time()
        if is_first_step:
            input_messages = [
                {
                    "role": MessageRole.USER,
                    "content": [
                        {
                            "type": "text",
                            "text": populate_template(
                                self.prompt_templates["planning"]["initial_plan"],
                                variables={"task": task, "tools": self.tools, "managed_agents": self.managed_agents},
                            ),
                        }
                    ],
                }
            ]
            if self.stream_outputs and hasattr(self.model, "generate_stream"):
                plan_message_content = ""
                output_stream = self.model.generate_stream(input_messages, stop_sequences=["<end_plan>"])  # type: ignore
                input_tokens, output_tokens = 0, 0
                with Live("", console=self.logger.console, vertical_overflow="visible") as live:
                    for event in output_stream:
                        if event.content is not None:
                            plan_message_content += event.content
                            live.update(Markdown(plan_message_content))
                            if event.token_usage:
                                output_tokens += event.token_usage.output_tokens
                                input_tokens = event.token_usage.input_tokens
                        yield event
            else:
                plan_message = self.model.generate(input_messages, stop_sequences=["<end_plan>"])
                plan_message_content = plan_message.content
                input_tokens, output_tokens = (
                    plan_message.token_usage.input_tokens,
                    plan_message.token_usage.output_tokens,
                )
            plan = textwrap.dedent(
                f"""Here are the facts I know and the plan of action that I will follow to solve the task:\n```\n{plan_message_content}\n```"""
            )
        else:
            # Summary mode removes the system prompt and previous planning messages output by the model.
            # Removing previous planning messages avoids influencing too much the new plan.
            memory_messages = self.write_memory_to_messages(summary_mode=True)
            plan_update_pre = {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"]["update_plan_pre_messages"], variables={"task": task}
                        ),
                    }
                ],
            }
            plan_update_post = {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"]["update_plan_post_messages"],
                            variables={
                                "task": task,
                                "tools": self.tools,
                                "managed_agents": self.managed_agents,
                                "remaining_steps": (self.max_steps - step),
                            },
                        ),
                    }
                ],
            }
            input_messages = [plan_update_pre] + memory_messages + [plan_update_post]
            if self.stream_outputs and hasattr(self.model, "generate_stream"):
                plan_message_content = ""
                input_tokens, output_tokens = 0, 0
                with Live("", console=self.logger.console, vertical_overflow="visible") as live:
                    for event in self.model.generate_stream(
                        input_messages,
                        stop_sequences=["<end_plan>"],
                    ):  # type: ignore
                        if event.content is not None:
                            plan_message_content += event.content
                            live.update(Markdown(plan_message_content))
                            if event.token_usage:
                                output_tokens += event.token_usage.output_tokens
                                input_tokens = event.token_usage.input_tokens
                        yield event
            else:
                plan_message = self.model.generate(input_messages, stop_sequences=["<end_plan>"])
                plan_message_content = plan_message.content
                input_tokens, output_tokens = (
                    plan_message.token_usage.input_tokens,
                    plan_message.token_usage.output_tokens,
                )
            plan = textwrap.dedent(
                f"""I still need to solve the task I was given:\n```\n{self.task}\n```\n\nHere are the facts I know and my new/updated plan of action to solve the task:\n```\n{plan_message_content}\n```"""
            )
        log_headline = "Initial plan" if is_first_step else "Updated plan"
        self.logger.log(Rule(f"[bold]{log_headline}", style="orange"), Text(plan), level=LogLevel.INFO)
        yield PlanningStep(
            model_input_messages=input_messages,
            plan=plan,
            model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content=plan_message_content),
            token_usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            timing=Timing(start_time=start_time, end_time=time.time()),
        )

    @property
    def logs(self):
        """
        获取代理日志 - 已弃用属性
        
        该属性用于获取代理的执行日志，包括系统提示和所有执行步骤。
        
        弃用说明:
            该属性已弃用，建议使用 self.memory.steps 替代。
            
        返回:
            list: 包含系统提示和所有执行步骤的列表
        """
        logger.warning(
            "The 'logs' attribute is deprecated and will soon be removed. Please use 'self.memory.steps' instead."
        )
        return [self.memory.system_prompt] + self.memory.steps

    @abstractmethod
    def initialize_system_prompt(self) -> str:
        """
        初始化系统提示词 - 抽象方法
        
        该方法必须在子类中实现，用于生成代理的系统提示词。
        系统提示词定义了代理的身份、能力和行为规范。
        
        返回:
            str: 格式化后的系统提示词字符串
            
        注意:
            这是一个抽象方法，子类必须实现此方法
        """
        ...

    def interrupt(self):
        """
        中断代理执行
        
        设置中断标志，使正在执行的代理在下一次循环检查时停止执行。
        这提供了一种优雅地停止长时间运行任务的方法。
        
        使用场景:
        - 用户主动取消任务
        - 超时控制
        - 错误恢复
        """
        self.interrupt_switch = True

    def write_memory_to_messages(
        self,
        summary_mode: bool | None = False,
    ) -> list[Message]:
        """
        将内存转换为消息列表
        
        从代理内存中读取过往的 LLM 输出、行动和观察或错误信息，
        转换为可用作 LLM 输入的消息序列。会添加关键词（如 PLAN、error 等）
        来帮助 LLM 更好地理解上下文。
        
        该方法是代理与 LLM 通信的关键桥梁，将结构化的内存数据
        转换为 LLM 可以理解的对话格式。
        
        参数:
            summary_mode (bool | None): 是否使用摘要模式
                - True: 生成简化的消息，去除部分详细信息
                - False: 生成完整的消息包含所有细节
                
        返回:
            list[Message]: 格式化的消息列表，可直接用作 LLM 输入
        """
        # 从系统提示开始构建消息列表
        messages = self.memory.system_prompt.to_messages(summary_mode=summary_mode)
        
        # 依次添加每个内存步骤的消息
        for memory_step in self.memory.steps:
            messages.extend(memory_step.to_messages(summary_mode=summary_mode))
        
        return messages

    def _step_stream(self, memory_step: ActionStep) -> Generator[ChatMessageStreamDelta | FinalOutput]:
        """
        执行单步 ReAct 框架流程 - 抽象方法
        
        该方法实现 ReAct 框架的单步执行：代理思考、行动并观察结果。
        如果启用流式输出，会在运行过程中产出 ChatMessageStreamDelta。
        
        ReAct 框架步骤:
        1. Reasoning（推理）：分析当前情况和已有信息
        2. Acting（行动）：选择并执行适当的工具或行动
        3. Observing（观察）：获取行动结果并更新理解
        
        参数:
            memory_step (ActionStep): 当前执行的行动步骤对象
            
        产出:
            Generator[ChatMessageStreamDelta | FinalOutput]: 
                - ChatMessageStreamDelta: 流式消息增量（如果启用流式输出）
                - FinalOutput: 最终输出（如果是最后一步，否则为 None）
                
        注意:
            这是一个抽象方法，子类必须实现此方法
        """
        raise NotImplementedError("This method should be implemented in child classes")

    def step(self, memory_step: ActionStep) -> Any:
        """
        执行单步 ReAct 框架流程 - 同步版本
        
        这是 _step_stream 的同步版本，执行单步 ReAct 框架流程：
        代理思考、行动并观察结果。返回最终结果而不是流式输出。
        
        该方法内部调用 _step_stream 并收集所有输出，返回最后一个元素
        （即最终答案或 None）。适合不需要流式输出的场景。
        
        参数:
            memory_step (ActionStep): 当前执行的行动步骤对象
            
        返回:
            Any: 如果是最后一步则返回最终答案，否则返回 None
        """
        return list(self._step_stream(memory_step))[-1]

    def extract_action(self, model_output: str, split_token: str) -> tuple[str, str]:
        """
        从 LLM 输出中解析行动
        
        解析 LLM 生成的文本，提取推理过程和具体行动。使用分隔符将
        模型输出分成推理部分和行动部分。
        
        解析策略:
        - 使用指定的分隔符分割文本
        - 从末尾开始索引，处理输出中可能包含多个分隔符的情况
        - 返回推理和行动两部分，并去除首尾空白
        
        参数:
            model_output (str): LLM 的原始输出文本
            split_token (str): 用于分割的标记，应与系统提示中的示例匹配
            
        返回:
            tuple[str, str]: (推理过程, 行动内容) 的元组
            
        异常:
            AgentParsingError: 当无法找到分隔符或解析失败时抛出
            
        示例:
            >>> output = "我需要搜索信息。Action: search('Python教程')"
            >>> rationale, action = extract_action(output, "Action:")
            >>> print(rationale)  # "我需要搜索信息。"
            >>> print(action)     # "search('Python教程')"
        """
        try:
            # 使用分隔符分割模型输出
            split = model_output.split(split_token)
            
            # 从末尾开始索引，处理输出中可能包含多个分隔符的情况
            # 倒数第二个部分是推理，最后一个部分是行动
            rationale, action = (
                split[-2],
                split[-1],
            )
        except Exception:
            # 解析失败时抛出详细的错误信息
            raise AgentParsingError(
                f"No '{split_token}' token provided in your output.\nYour output:\n{model_output}\n. Be sure to include an action, prefaced with '{split_token}'!",
                self.logger,
            )
        
        # 去除首尾空白并返回
        return rationale.strip(), action.strip()

    def provide_final_answer(self, task: str, images: list["PIL.Image.Image"] | None = None) -> ChatMessage:
        """
        基于代理交互日志提供任务的最终答案
        
        当代理达到最大步数或需要强制结束时，使用该方法生成最终答案。
        该方法会构造特殊的消息序列，包含最终答案提示模板和完整的执行历史。
        
        消息构造流程:
        1. 添加最终答案前置系统提示
        2. 如果有图像，添加图像内容
        3. 添加完整的代理执行历史（除了系统提示）
        4. 添加最终答案后置用户提示
        5. 调用模型生成最终答案
        
        参数:
            task (str): 要执行的任务描述
            images (list["PIL.Image.Image"] | None): 可选的图像对象列表
            
        返回:
            ChatMessage: 包含最终答案的聊天消息对象
            如果生成失败，返回包含错误信息的字符串
            
        注意:
            该方法主要用于异常情况下的最终答案生成，
            正常情况下代理应该通过 final_answer 工具提供答案
        """
        # 构建最终答案消息序列
        messages = [
            {
                "role": MessageRole.SYSTEM,
                "content": [
                    {
                        "type": "text",
                        "text": self.prompt_templates["final_answer"]["pre_messages"],
                    }
                ],
            }
        ]
        
        # 如果提供了图像，添加图像内容
        if images:
            messages[0]["content"].append({"type": "image"})
        
        # 添加代理的执行历史（跳过系统提示，从索引1开始）
        messages += self.write_memory_to_messages()[1:]
        
        # 添加最终答案后置提示
        messages += [
            {
                "role": MessageRole.USER,
                "content": [
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["final_answer"]["post_messages"], variables={"task": task}
                        ),
                    }
                ],
            }
        ]
        
        try:
            # 调用模型生成最终答案
            chat_message: ChatMessage = self.model.generate(messages)
            return chat_message
        except Exception as e:
            # 生成失败时返回错误信息
            return f"Error in generating final LLM output:\n{e}"

    def visualize(self):
        """
        创建代理结构的富文本树形可视化
        
        生成一个树状图显示代理的结构，包括：
        - 代理的基本信息（名称、类型等）
        - 可用工具列表
        - 管理的子代理
        - 配置信息
        
        该方法有助于理解复杂代理系统的层次结构和组件关系。
        可视化结果会通过日志记录器输出到控制台。
        
        使用场景:
        - 调试代理配置
        - 了解代理能力
        - 文档生成
        - 系统架构展示
        """
        self.logger.visualize_agent_tree(self)

    def replay(self, detailed: bool = False):
        """
        重放代理执行步骤的美观展示
        
        按时间顺序重新显示代理的所有执行步骤，包括：
        - 系统提示信息
        - 每个步骤的输入和输出
        - 工具调用和结果
        - 错误信息（如果有）
        - 最终答案
        
        重放功能有助于：
        - 理解代理的决策过程
        - 调试执行问题
        - 分析性能瓶颈
        - 验证执行逻辑
        
        参数:
            detailed (bool): 是否显示详细信息
                - False: 只显示关键步骤和结果
                - True: 显示每步的完整内存状态
                
        警告:
            启用详细模式会显著增加输出长度，建议仅在调试时使用
        """
        self.memory.replay(self.logger, detailed=detailed)

    def __call__(self, task: str, **kwargs):
        """
        作为子代理被调用时的处理方法
        
        当该代理作为另一个代理的子代理被调用时，此方法会被执行。
        它为子代理添加额外的提示信息，运行任务，并包装输出结果。
        
        处理流程:
        1. 使用管理代理模板构造完整任务描述
        2. 运行代理执行任务
        3. 提取执行结果
        4. 使用报告模板格式化输出
        5. 可选地添加详细执行摘要
        
        参数:
            task (str): 分配给子代理的任务描述
            **kwargs: 传递给 run 方法的额外参数
            
        返回:
            str: 格式化后的执行报告，包含：
                - 任务执行结果
                - 代理身份信息
                - 可选的详细执行摘要
                
        注意:
            该方法只会在代理作为子代理被其他代理调用时使用，
            不应直接在用户代码中调用
        """
        # 使用管理代理任务模板构造完整任务
        full_task = populate_template(
            self.prompt_templates["managed_agent"]["task"],
            variables=dict(name=self.name, task=task),
        )
        
        # 运行代理执行任务
        result = self.run(full_task, **kwargs)
        
        # 提取执行结果
        if isinstance(result, RunResult):
            report = result.output
        else:
            report = result
            
        # 使用报告模板格式化输出
        answer = populate_template(
            self.prompt_templates["managed_agent"]["report"], 
            variables=dict(name=self.name, final_answer=report)
        )
        
        # 如果配置了提供运行摘要，添加详细信息
        if self.provide_run_summary:
            answer += "\n\nFor more detail, find below a summary of this agent's work:\n<summary_of_work>\n"
            # 添加执行历史摘要
            for message in self.write_memory_to_messages(summary_mode=True):
                content = message["content"]
                answer += "\n" + truncate_content(str(content)) + "\n---"
            answer += "\n</summary_of_work>"
            
        return answer

    def save(self, output_dir: str | Path, relative_path: str | None = None):
        """
        Saves the relevant code files for your agent. This will copy the code of your agent in `output_dir` as well as autogenerate:

        - a `tools` folder containing the logic for each of the tools under `tools/{tool_name}.py`.
        - a `managed_agents` folder containing the logic for each of the managed agents.
        - an `agent.json` file containing a dictionary representing your agent.
        - a `prompt.yaml` file containing the prompt templates used by your agent.
        - an `app.py` file providing a UI for your agent when it is exported to a Space with `agent.push_to_hub()`
        - a `requirements.txt` containing the names of the modules used by your tool (as detected when inspecting its
          code)

        Args:
            output_dir (`str` or `Path`): The folder in which you want to save your agent.
        """
        make_init_file(output_dir)

        # Recursively save managed agents
        if self.managed_agents:
            make_init_file(os.path.join(output_dir, "managed_agents"))
            for agent_name, agent in self.managed_agents.items():
                agent_suffix = f"managed_agents.{agent_name}"
                if relative_path:
                    agent_suffix = relative_path + "." + agent_suffix
                agent.save(os.path.join(output_dir, "managed_agents", agent_name), relative_path=agent_suffix)

        class_name = self.__class__.__name__

        # Save tools to different .py files
        for tool in self.tools.values():
            make_init_file(os.path.join(output_dir, "tools"))
            tool.save(os.path.join(output_dir, "tools"), tool_file_name=tool.name, make_gradio_app=False)

        # Save prompts to yaml
        yaml_prompts = yaml.safe_dump(
            self.prompt_templates,
            default_style="|",  # This forces block literals for all strings
            default_flow_style=False,
            width=float("inf"),
            sort_keys=False,
            allow_unicode=True,
            indent=2,
        )

        with open(os.path.join(output_dir, "prompts.yaml"), "w", encoding="utf-8") as f:
            f.write(yaml_prompts)

        # Save agent dictionary to json
        agent_dict = self.to_dict()
        agent_dict["tools"] = [tool.name for tool in self.tools.values()]
        agent_dict["managed_agents"] = {agent.name: agent.__class__.__name__ for agent in self.managed_agents.values()}
        with open(os.path.join(output_dir, "agent.json"), "w", encoding="utf-8") as f:
            json.dump(agent_dict, f, indent=4)

        # Save requirements
        with open(os.path.join(output_dir, "requirements.txt"), "w", encoding="utf-8") as f:
            f.writelines(f"{r}\n" for r in agent_dict["requirements"])

        # Make agent.py file with Gradio UI
        agent_name = f"agent_{self.name}" if getattr(self, "name", None) else "agent"
        managed_agent_relative_path = relative_path + "." if relative_path is not None else ""
        app_template = textwrap.dedent("""
            import yaml
            import os
            from smolagents import GradioUI, {{ class_name }}, {{ agent_dict['model']['class'] }}

            # Get current directory path
            CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

            {% for tool in tools.values() -%}
            from {{managed_agent_relative_path}}tools.{{ tool.name }} import {{ tool.__class__.__name__ }} as {{ tool.name | camelcase }}
            {% endfor %}
            {% for managed_agent in managed_agents.values() -%}
            from {{managed_agent_relative_path}}managed_agents.{{ managed_agent.name }}.app import agent_{{ managed_agent.name }}
            {% endfor %}

            model = {{ agent_dict['model']['class'] }}(
            {% for key in agent_dict['model']['data'] if key not in ['class', 'last_input_token_count', 'last_output_token_count'] -%}
                {{ key }}={{ agent_dict['model']['data'][key]|repr }},
            {% endfor %})

            {% for tool in tools.values() -%}
            {{ tool.name }} = {{ tool.name | camelcase }}()
            {% endfor %}

            with open(os.path.join(CURRENT_DIR, "prompts.yaml"), 'r') as stream:
                prompt_templates = yaml.safe_load(stream)

            {{ agent_name }} = {{ class_name }}(
                model=model,
                tools=[{% for tool_name in tools.keys() if tool_name != "final_answer" %}{{ tool_name }}{% if not loop.last %}, {% endif %}{% endfor %}],
                managed_agents=[{% for subagent_name in managed_agents.keys() %}agent_{{ subagent_name }}{% if not loop.last %}, {% endif %}{% endfor %}],
                {% for attribute_name, value in agent_dict.items() if attribute_name not in ["model", "tools", "prompt_templates", "authorized_imports", "managed_agents", "requirements"] -%}
                {{ attribute_name }}={{ value|repr }},
                {% endfor %}prompt_templates=prompt_templates
            )
            if __name__ == "__main__":
                GradioUI({{ agent_name }}).launch()
            """).strip()
        template_env = jinja2.Environment(loader=jinja2.BaseLoader(), undefined=jinja2.StrictUndefined)
        template_env.filters["repr"] = repr
        template_env.filters["camelcase"] = lambda value: "".join(word.capitalize() for word in value.split("_"))
        template = template_env.from_string(app_template)

        # Render the app.py file from Jinja2 template
        app_text = template.render(
            {
                "agent_name": agent_name,
                "class_name": class_name,
                "agent_dict": agent_dict,
                "tools": self.tools,
                "managed_agents": self.managed_agents,
                "managed_agent_relative_path": managed_agent_relative_path,
            }
        )

        with open(os.path.join(output_dir, "app.py"), "w", encoding="utf-8") as f:
            f.write(app_text + "\n")  # Append newline at the end

    def to_dict(self) -> dict[str, Any]:
        """
        将代理转换为字典表示形式
        
        该方法将代理的所有配置和组件序列化为字典格式，便于：
        - 保存代理配置到文件
        - 网络传输代理定义
        - 版本控制和备份
        - 跨平台部署
        
        序列化内容包括:
        - 代理类名和基本配置
        - 工具列表和依赖项
        - 模型配置
        - 子代理配置
        - 提示模板
        - 执行参数
        
        返回:
            dict[str, Any]: 代理的字典表示，包含所有必要的配置信息
            
        注意:
            - step_callbacks 和 final_answer_checks 无法序列化，会被忽略
            - 返回的字典可用于 from_dict 方法重新创建代理
        """
        # 警告用户无法序列化的属性
        for attr in ["final_answer_checks", "step_callbacks"]:
            if getattr(self, attr, None):
                self.logger.log(f"This agent has {attr}: they will be ignored by this method.", LogLevel.INFO)

        # 序列化所有工具
        tool_dicts = [tool.to_dict() for tool in self.tools.values()]
        
        # 收集工具的依赖项
        tool_requirements = {req for tool in self.tools.values() for req in tool.to_dict()["requirements"]}
        
        # 收集管理代理的依赖项
        managed_agents_requirements = {
            req for managed_agent in self.managed_agents.values() for req in managed_agent.to_dict()["requirements"]
        }
        
        # 合并所有依赖项
        requirements = tool_requirements | managed_agents_requirements
        
        # 添加授权导入的依赖项（如果存在）
        if hasattr(self, "authorized_imports"):
            requirements.update(
                {package.split(".")[0] for package in self.authorized_imports if package not in BASE_BUILTIN_MODULES}
            )

        # 构建完整的代理字典
        agent_dict = {
            "class": self.__class__.__name__,  # 代理类名
            "tools": tool_dicts,  # 工具配置列表
            "model": {  # 模型配置
                "class": self.model.__class__.__name__,
                "data": self.model.to_dict(),
            },
            "managed_agents": [managed_agent.to_dict() for managed_agent in self.managed_agents.values()],  # 子代理配置
            "prompt_templates": self.prompt_templates,  # 提示模板
            "max_steps": self.max_steps,  # 最大步数
            "verbosity_level": int(self.logger.level),  # 日志级别
            "grammar": self.grammar,  # 语法规则（已弃用）
            "planning_interval": self.planning_interval,  # 规划间隔
            "name": self.name,  # 代理名称
            "description": self.description,  # 代理描述
            "requirements": sorted(requirements),  # 依赖项列表
        }
        return agent_dict

    @classmethod
    def from_dict(cls, agent_dict: dict[str, Any], **kwargs) -> "MultiStepAgent":
        """Create agent from a dictionary representation.

        Args:
            agent_dict (`dict[str, Any]`): Dictionary representation of the agent.
            **kwargs: Additional keyword arguments that will override agent_dict values.

        Returns:
            `MultiStepAgent`: Instance of the agent class.
        """
        # Load model
        model_info = agent_dict["model"]
        model_class = getattr(importlib.import_module("smolagents.models"), model_info["class"])
        model = model_class.from_dict(model_info["data"])
        # Load tools
        tools = []
        for tool_info in agent_dict["tools"]:
            tools.append(Tool.from_code(tool_info["code"]))
        # Load managed agents
        managed_agents = []
        for managed_agent_name, managed_agent_class_name in agent_dict["managed_agents"].items():
            managed_agent_class = getattr(importlib.import_module("smolagents.agents"), managed_agent_class_name)
            managed_agents.append(managed_agent_class.from_dict(agent_dict["managed_agents"][managed_agent_name]))
        # Extract base agent parameters
        agent_args = {
            "model": model,
            "tools": tools,
            "prompt_templates": agent_dict.get("prompt_templates"),
            "max_steps": agent_dict.get("max_steps"),
            "verbosity_level": agent_dict.get("verbosity_level"),
            "grammar": agent_dict.get("grammar"),
            "planning_interval": agent_dict.get("planning_interval"),
            "name": agent_dict.get("name"),
            "description": agent_dict.get("description"),
        }
        # Filter out None values to use defaults from __init__
        agent_args = {k: v for k, v in agent_args.items() if v is not None}
        # Update with any additional kwargs
        agent_args.update(kwargs)
        # Create agent instance
        return cls(**agent_args)

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        token: str | None = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """
        Loads an agent defined on the Hub.

        <Tip warning={true}>

        Loading a tool from the Hub means that you'll download the tool and execute it locally.
        ALWAYS inspect the tool you're downloading before loading it within your runtime, as you would do when
        installing a package using pip/npm/apt.

        </Tip>

        Args:
            repo_id (`str`):
                The name of the repo on the Hub where your tool is defined.
            token (`str`, *optional*):
                The token to identify you on hf.co. If unset, will use the token generated when running
                `huggingface-cli login` (stored in `~/.huggingface`).
            trust_remote_code(`bool`, *optional*, defaults to False):
                This flags marks that you understand the risk of running remote code and that you trust this tool.
                If not setting this to True, loading the tool from Hub will fail.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments that will be split in two: all arguments relevant to the Hub (such as
                `cache_dir`, `revision`, `subfolder`) will be used when downloading the files for your agent, and the
                others will be passed along to its init.
        """
        if not trust_remote_code:
            raise ValueError(
                "Loading an agent from Hub requires to acknowledge you trust its code: to do so, pass `trust_remote_code=True`."
            )

        # Get the agent's Hub folder.
        download_kwargs = {"token": token, "repo_type": "space"} | {
            key: kwargs.pop(key)
            for key in [
                "cache_dir",
                "force_download",
                "proxies",
                "revision",
                "local_files_only",
            ]
            if key in kwargs
        }

        download_folder = Path(snapshot_download(repo_id=repo_id, **download_kwargs))
        return cls.from_folder(download_folder, **kwargs)

    @classmethod
    def from_folder(cls, folder: str | Path, **kwargs):
        """Loads an agent from a local folder.

        Args:
            folder (`str` or `Path`): The folder where the agent is saved.
            **kwargs: Additional keyword arguments that will be passed to the agent's init.
        """
        # Load agent.json
        folder = Path(folder)
        agent_dict = json.loads((folder / "agent.json").read_text())

        # Load managed agents from their respective folders, recursively
        managed_agents = []
        for managed_agent_name, managed_agent_class_name in agent_dict["managed_agents"].items():
            agent_cls = getattr(importlib.import_module("smolagents.agents"), managed_agent_class_name)
            managed_agents.append(agent_cls.from_folder(folder / "managed_agents" / managed_agent_name))
        agent_dict["managed_agents"] = {}

        # Load tools
        tools = []
        for tool_name in agent_dict["tools"]:
            tool_code = (folder / "tools" / f"{tool_name}.py").read_text()
            tools.append({"name": tool_name, "code": tool_code})
        agent_dict["tools"] = tools

        # Add managed agents to kwargs to override the empty list in from_dict
        if managed_agents:
            kwargs["managed_agents"] = managed_agents

        return cls.from_dict(agent_dict, **kwargs)

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload agent",
        private: bool | None = None,
        token: bool | str | None = None,
        create_pr: bool = False,
    ) -> str:
        """
        Upload the agent to the Hub.

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push to. It should contain your organization name when
                pushing to a given organization.
            commit_message (`str`, *optional*, defaults to `"Upload agent"`):
                Message to commit while pushing.
            private (`bool`, *optional*, defaults to `None`):
                Whether to make the repo private. If `None`, the repo will be public unless the organization's default is private. This value is ignored if the repo already exists.
            token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether to create a PR with the uploaded files or directly commit.
        """
        repo_url = create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=True,
            repo_type="space",
            space_sdk="gradio",
        )
        repo_id = repo_url.repo_id
        metadata_update(
            repo_id,
            {"tags": ["smolagents", "agent"]},
            repo_type="space",
            token=token,
            overwrite=True,
        )

        with tempfile.TemporaryDirectory() as work_dir:
            self.save(work_dir)
            logger.info(f"Uploading the following files to {repo_id}: {','.join(os.listdir(work_dir))}")
            return upload_folder(
                repo_id=repo_id,
                commit_message=commit_message,
                folder_path=work_dir,
                token=token,
                create_pr=create_pr,
                repo_type="space",
            )


class ToolCallingAgent(MultiStepAgent):
    """
    工具调用代理 - 使用结构化工具调用的智能代理
    
    该代理使用类似 JSON 的工具调用格式，利用 `model.get_tool_call` 方法
    来发挥 LLM 引擎的工具调用能力。相比 CodeAgent，该代理更加结构化，
    适合需要明确工具调用界面的场景。
    
    主要特点:
    - 使用结构化的 JSON 格式进行工具调用
    - 支持流式输出
    - 更好的工具调用错误处理
    - 支持状态变量替换
    
    参数:
        tools (`list[Tool]`): 代理可以使用的工具列表
        model (`Model`): 生成代理行动的模型
        prompt_templates ([`~agents.PromptTemplates`], *可选*): 提示模板集合
        planning_interval (`int`, *可选*): 执行规划步骤的间隔
        stream_outputs (`bool`, *可选*, 默认 `False`): 执行期间是否流式输出
        **kwargs: 其他关键字参数
    """

    def __init__(
        self,
        tools: list[Tool],
        model: Model,
        prompt_templates: PromptTemplates | None = None,
        planning_interval: int | None = None,
        stream_outputs: bool = False,
        **kwargs,
    ):
        prompt_templates = prompt_templates or yaml.safe_load(
            importlib.resources.files("smolagents.prompts").joinpath("toolcalling_agent.yaml").read_text()
        )
        super().__init__(
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            planning_interval=planning_interval,
            **kwargs,
        )

        # Streaming setup
        self.stream_outputs = stream_outputs
        if self.stream_outputs and not hasattr(self.model, "generate_stream"):
            raise ValueError(
                "`stream_outputs` is set to True, but the model class implements no `generate_stream` method."
            )

    def initialize_system_prompt(self) -> str:
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={"tools": self.tools, "managed_agents": self.managed_agents},
        )
        return system_prompt

    def _step_stream(self, memory_step: ActionStep) -> Generator[ChatMessageStreamDelta | FinalOutput]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Yields ChatMessageStreamDelta during the run if streaming is enabled.
        At the end, yields either None if the step is not final, or the final answer.
        """
        memory_messages = self.write_memory_to_messages()

        input_messages = memory_messages.copy()

        # Add new step in logs
        memory_step.model_input_messages = input_messages

        try:
            if self.stream_outputs and hasattr(self.model, "generate_stream"):
                output_stream = self.model.generate_stream(
                    input_messages,
                    stop_sequences=["Observation:", "Calling tools:"],
                    tools_to_call_from=list(self.tools.values()),
                )

                model_output = ""
                input_tokens, output_tokens = 0, 0
                tool_calls = {}

                with Live("", console=self.logger.console, vertical_overflow="visible") as live:
                    for event in output_stream:
                        if event.content is not None:
                            model_output += event.content
                            if event.token_usage:
                                output_tokens += event.token_usage.output_tokens
                                input_tokens = event.token_usage.input_tokens
                        if event.tool_calls:
                            tool_calls.update({tool_call.id: tool_call for tool_call in event.tool_calls})
                        # Propagate the streaming delta
                        live.update(
                            Markdown(model_output + "\n".join([str(tool_call) for tool_call in tool_calls.values()]))
                        )
                        yield event

                chat_message = ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=model_output,
                    token_usage=TokenUsage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    ),
                    tool_calls=list(tool_calls.values()),
                )
            else:
                chat_message: ChatMessage = self.model.generate(
                    input_messages,
                    stop_sequences=["Observation:", "Calling tools:"],
                    tools_to_call_from=list(self.tools.values()),
                )

                model_output = chat_message.content
                self.logger.log_markdown(
                    content=model_output if model_output else str(chat_message.raw),
                    title="Output message of the LLM:",
                    level=LogLevel.DEBUG,
                )

            # Record model output
            memory_step.model_output_message = chat_message
            memory_step.model_output = model_output
        except Exception as e:
            raise AgentGenerationError(f"Error while generating output:\n{e}", self.logger) from e

        if chat_message.tool_calls is None or len(chat_message.tool_calls) == 0:
            try:
                chat_message = self.model.parse_tool_calls(chat_message)
            except Exception as e:
                raise AgentParsingError(f"Error while parsing tool call from model output: {e}", self.logger)
        else:
            for tool_call in chat_message.tool_calls:
                tool_call.function.arguments = parse_json_if_needed(tool_call.function.arguments)
        tool_call = chat_message.tool_calls[0]  # type: ignore
        tool_name, tool_call_id = tool_call.function.name, tool_call.id
        tool_arguments = tool_call.function.arguments
        memory_step.model_output = str(f"Called Tool: '{tool_name}' with arguments: {tool_arguments}")
        memory_step.tool_calls = [ToolCall(name=tool_name, arguments=tool_arguments, id=tool_call_id)]
        memory_step.token_usage = chat_message.token_usage

        # Execute
        self.logger.log(
            Panel(Text(f"Calling tool: '{tool_name}' with arguments: {tool_arguments}")),
            level=LogLevel.INFO,
        )
        if tool_name == "final_answer":
            answer = (
                tool_arguments["answer"]
                if isinstance(tool_arguments, dict) and "answer" in tool_arguments
                else tool_arguments
            )
            if isinstance(answer, str) and answer in self.state.keys():
                # if the answer is a state variable, return the value
                # State variables are not JSON-serializable (AgentImage, AgentAudio) so can't be passed as arguments to execute_tool_call
                final_answer = self.state[answer]
                self.logger.log(
                    f"[bold {YELLOW_HEX}]Final answer:[/bold {YELLOW_HEX}] Extracting key '{answer}' from state to return value '{final_answer}'.",
                    level=LogLevel.INFO,
                )
            else:
                # Allow arbitrary keywords
                final_answer = self.execute_tool_call("final_answer", tool_arguments)
                self.logger.log(
                    Text(f"Final answer: {final_answer}", style=f"bold {YELLOW_HEX}"),
                    level=LogLevel.INFO,
                )

            memory_step.action_output = final_answer
            yield FinalOutput(output=final_answer)
        else:
            if tool_arguments is None:
                tool_arguments = {}
            observation = self.execute_tool_call(tool_name, tool_arguments)
            observation_type = type(observation)
            if observation_type in [AgentImage, AgentAudio]:
                if observation_type == AgentImage:
                    observation_name = "image.png"
                elif observation_type == AgentAudio:
                    observation_name = "audio.mp3"
                # TODO: observation naming could allow for different names of same type

                self.state[observation_name] = observation
                updated_information = f"Stored '{observation_name}' in memory."
            else:
                updated_information = str(observation).strip()
            self.logger.log(
                f"Observations: {updated_information.replace('[', '|')}",  # escape potential rich-tag-like components
                level=LogLevel.INFO,
            )
            memory_step.observations = updated_information
            yield FinalOutput(output=None)

    def _substitute_state_variables(self, arguments: dict[str, str] | str) -> dict[str, Any] | str:
        """Replace string values in arguments with their corresponding state values if they exist."""
        if isinstance(arguments, dict):
            return {
                key: self.state.get(value, value) if isinstance(value, str) else value
                for key, value in arguments.items()
            }
        return arguments

    def execute_tool_call(self, tool_name: str, arguments: dict[str, str] | str) -> Any:
        """
        Execute a tool or managed agent with the provided arguments.

        The arguments are replaced with the actual values from the state if they refer to state variables.

        Args:
            tool_name (`str`): Name of the tool or managed agent to execute.
            arguments (dict[str, str] | str): Arguments passed to the tool call.
        """
        # Check if the tool exists
        available_tools = {**self.tools, **self.managed_agents}
        if tool_name not in available_tools:
            raise AgentToolExecutionError(
                f"Unknown tool {tool_name}, should be one of: {', '.join(available_tools)}.", self.logger
            )

        # Get the tool and substitute state variables in arguments
        tool = available_tools[tool_name]
        arguments = self._substitute_state_variables(arguments)
        is_managed_agent = tool_name in self.managed_agents

        try:
            # Call tool with appropriate arguments
            if isinstance(arguments, dict):
                return tool(**arguments) if is_managed_agent else tool(**arguments, sanitize_inputs_outputs=True)
            elif isinstance(arguments, str):
                return tool(arguments) if is_managed_agent else tool(arguments, sanitize_inputs_outputs=True)
            else:
                raise TypeError(f"Unsupported arguments type: {type(arguments)}")

        except TypeError as e:
            # Handle invalid arguments
            description = getattr(tool, "description", "No description")
            if is_managed_agent:
                error_msg = (
                    f"Invalid request to team member '{tool_name}' with arguments {json.dumps(arguments)}: {e}\n"
                    "You should call this team member with a valid request.\n"
                    f"Team member description: {description}"
                )
            else:
                error_msg = (
                    f"Invalid call to tool '{tool_name}' with arguments {json.dumps(arguments)}: {e}\n"
                    "You should call this tool with correct input arguments.\n"
                    f"Expected inputs: {json.dumps(tool.inputs)}\n"
                    f"Returns output type: {tool.output_type}\n"
                    f"Tool description: '{description}'"
                )
            raise AgentToolCallError(error_msg, self.logger) from e

        except Exception as e:
            # Handle execution errors
            if is_managed_agent:
                error_msg = (
                    f"Error executing request to team member '{tool_name}' with arguments {json.dumps(arguments)}: {e}\n"
                    "Please try again or request to another team member"
                )
            else:
                error_msg = (
                    f"Error executing tool '{tool_name}' with arguments {json.dumps(arguments)}: {type(e).__name__}: {e}\n"
                    "Please try again or use another tool"
                )
            raise AgentToolExecutionError(error_msg, self.logger) from e


class CodeAgent(MultiStepAgent):
    """
    代码执行代理 - 通过生成和执行代码来解决任务的智能代理
    
    该代理中，工具调用将由 LLM 以代码格式制定，然后被解析和执行。
    这是一个非常强大的代理类型，能够处理复杂的数据分析、计算和操作任务。
    
    主要特点:
    - 支持 Python 代码生成和执行
    - 支持多种执行环境（本地、Docker、E2B）
    - 支持自定义导入模块
    - 支持结构化输出生成
    - 强大的错误处理和安全控制
    
    参数:
        tools (`list[Tool]`): 代理可以使用的工具列表
        model (`Model`): 生成代理行动的模型
        prompt_templates ([`~agents.PromptTemplates`], *可选*): 提示模板集合
        additional_authorized_imports (`list[str]`, *可选*): 代理的额外授权导入模块
        planning_interval (`int`, *可选*): 执行规划步骤的间隔
        executor_type (`str`, 默认 `"local"`): 使用的执行器类型，可选 `"local"`、`"e2b"` 或 `"docker"`
        executor_kwargs (`dict`, *可选*): 初始化执行器时传递的额外参数
        max_print_outputs_length (`int`, *可选*): 打印输出的最大长度
        stream_outputs (`bool`, *可选*, 默认 `False`): 执行期间是否流式输出
        use_structured_outputs_internally (`bool`, 默认 `False`): 是否在每个行动步骤使用结构化生成：
            可以提高许多模型的性能
            <新增 版本="1.17.0"/>
        grammar (`dict[str, str]`, *可选*): 用于解析 LLM 输出的语法规则
            <已弃用 版本="1.17.0">
            参数 `grammar` 已弃用，将在版本 1.20 中移除。
            </已弃用>
        **kwargs: 其他关键字参数
    """

    def __init__(
        self,
        tools: list[Tool],
        model: Model,
        prompt_templates: PromptTemplates | None = None,
        additional_authorized_imports: list[str] | None = None,
        planning_interval: int | None = None,
        executor_type: str | None = "local",
        executor_kwargs: dict[str, Any] | None = None,
        max_print_outputs_length: int | None = None,
        stream_outputs: bool = False,
        use_structured_outputs_internally: bool = False,
        grammar: dict[str, str] | None = None,
        **kwargs,
    ):
        self.additional_authorized_imports = additional_authorized_imports if additional_authorized_imports else []
        self.authorized_imports = sorted(set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))
        self.max_print_outputs_length = max_print_outputs_length
        self._use_structured_outputs_internally = use_structured_outputs_internally
        if use_structured_outputs_internally:
            prompt_templates = prompt_templates or yaml.safe_load(
                importlib.resources.files("smolagents.prompts").joinpath("structured_code_agent.yaml").read_text()
            )
        else:
            prompt_templates = prompt_templates or yaml.safe_load(
                importlib.resources.files("smolagents.prompts").joinpath("code_agent.yaml").read_text()
            )
        if grammar and use_structured_outputs_internally:
            raise ValueError("You cannot use 'grammar' and 'use_structured_outputs_internally' at the same time.")
        super().__init__(
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            grammar=grammar,
            planning_interval=planning_interval,
            **kwargs,
        )
        self.stream_outputs = stream_outputs
        if self.stream_outputs and not hasattr(self.model, "generate_stream"):
            raise ValueError(
                "`stream_outputs` is set to True, but the model class implements no `generate_stream` method."
            )
        if "*" in self.additional_authorized_imports:
            self.logger.log(
                "Caution: you set an authorization for all imports, meaning your agent can decide to import any package it deems necessary. This might raise issues if the package is not installed in your environment.",
                level=LogLevel.INFO,
            )
        self.executor_type = executor_type or "local"
        self.executor_kwargs = executor_kwargs or {}
        self.python_executor = self.create_python_executor()

    def create_python_executor(self) -> PythonExecutor:
        match self.executor_type:
            case "e2b" | "docker":
                if self.managed_agents:
                    raise Exception("Managed agents are not yet supported with remote code execution.")
                if self.executor_type == "e2b":
                    return E2BExecutor(self.additional_authorized_imports, self.logger, **self.executor_kwargs)
                else:
                    return DockerExecutor(self.additional_authorized_imports, self.logger, **self.executor_kwargs)
            case "local":
                return LocalPythonExecutor(
                    self.additional_authorized_imports,
                    **{"max_print_outputs_length": self.max_print_outputs_length} | self.executor_kwargs,
                )
            case _:  # if applicable
                raise ValueError(f"Unsupported executor type: {self.executor_type}")

    def initialize_system_prompt(self) -> str:
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={
                "tools": self.tools,
                "managed_agents": self.managed_agents,
                "authorized_imports": (
                    "You can import from any package you want."
                    if "*" in self.authorized_imports
                    else str(self.authorized_imports)
                ),
            },
        )
        return system_prompt

    def _step_stream(self, memory_step: ActionStep) -> Generator[ChatMessageStreamDelta | FinalOutput]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Yields ChatMessageStreamDelta during the run if streaming is enabled.
        At the end, yields either None if the step is not final, or the final answer.
        """
        memory_messages = self.write_memory_to_messages()

        input_messages = memory_messages.copy()
        ### Generate model output ###
        memory_step.model_input_messages = input_messages
        try:
            additional_args: dict[str, Any] = {}
            if self.grammar:
                additional_args["grammar"] = self.grammar
            if self._use_structured_outputs_internally:
                additional_args["response_format"] = CODEAGENT_RESPONSE_FORMAT
            if self.stream_outputs:
                output_stream = self.model.generate_stream(
                    input_messages,
                    stop_sequences=["<end_code>", "Observation:", "Calling tools:"],
                    **additional_args,
                )
                output_text = ""
                input_tokens, output_tokens = 0, 0
                with Live("", console=self.logger.console, vertical_overflow="visible") as live:
                    for event in output_stream:
                        if event.content is not None:
                            output_text += event.content
                            live.update(Markdown(output_text))
                            if event.token_usage:
                                output_tokens += event.token_usage.output_tokens
                                input_tokens = event.token_usage.input_tokens
                        assert isinstance(event, ChatMessageStreamDelta)
                        yield event

                chat_message = ChatMessage(
                    role="assistant",
                    content=output_text,
                    token_usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
                )
                memory_step.model_output_message = chat_message
                output_text = chat_message.content
            else:
                chat_message: ChatMessage = self.model.generate(
                    input_messages,
                    stop_sequences=["<end_code>", "Observation:", "Calling tools:"],
                    **additional_args,
                )
                memory_step.model_output_message = chat_message
                output_text = chat_message.content
                self.logger.log_markdown(
                    content=output_text,
                    title="Output message of the LLM:",
                    level=LogLevel.DEBUG,
                )

            # This adds <end_code> sequence to the history.
            # This will nudge ulterior LLM calls to finish with <end_code>, thus efficiently stopping generation.
            if output_text and output_text.strip().endswith("```"):
                output_text += "<end_code>"
                memory_step.model_output_message.content = output_text

            memory_step.token_usage = chat_message.token_usage
            memory_step.model_output = output_text
        except Exception as e:
            raise AgentGenerationError(f"Error in generating model output:\n{e}", self.logger) from e

        ### Parse output ###
        try:
            if self._use_structured_outputs_internally:
                code_action = json.loads(output_text)["code"]
                code_action = extract_code_from_text(code_action) or code_action
            else:
                code_action = parse_code_blobs(output_text)
            code_action = fix_final_answer_code(code_action)
        except Exception as e:
            error_msg = f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
            raise AgentParsingError(error_msg, self.logger)

        memory_step.tool_calls = [
            ToolCall(
                name="python_interpreter",
                arguments=code_action,
                id=f"call_{len(self.memory.steps)}",
            )
        ]

        ### Execute action ###
        self.logger.log_code(title="Executing parsed code:", content=code_action, level=LogLevel.INFO)
        is_final_answer = False
        try:
            output, execution_logs, is_final_answer = self.python_executor(code_action)
            execution_outputs_console = []
            if len(execution_logs) > 0:
                execution_outputs_console += [
                    Text("Execution logs:", style="bold"),
                    Text(execution_logs),
                ]
            observation = "Execution logs:\n" + execution_logs
        except Exception as e:
            if hasattr(self.python_executor, "state") and "_print_outputs" in self.python_executor.state:
                execution_logs = str(self.python_executor.state["_print_outputs"])
                if len(execution_logs) > 0:
                    execution_outputs_console = [
                        Text("Execution logs:", style="bold"),
                        Text(execution_logs),
                    ]
                    memory_step.observations = "Execution logs:\n" + execution_logs
                    self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
            error_msg = str(e)
            if "Import of " in error_msg and " is not allowed" in error_msg:
                self.logger.log(
                    "[bold red]Warning to user: Code execution failed due to an unauthorized import - Consider passing said import under `additional_authorized_imports` when initializing your CodeAgent.",
                    level=LogLevel.INFO,
                )
            raise AgentExecutionError(error_msg, self.logger)

        truncated_output = truncate_content(str(output))
        observation += "Last output from code snippet:\n" + truncated_output
        memory_step.observations = observation

        execution_outputs_console += [
            Text(
                f"{('Out - Final answer' if is_final_answer else 'Out')}: {truncated_output}",
                style=(f"bold {YELLOW_HEX}" if is_final_answer else ""),
            ),
        ]
        self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
        memory_step.action_output = output
        yield FinalOutput(output=output if is_final_answer else None)

    def to_dict(self) -> dict[str, Any]:
        """Convert the agent to a dictionary representation.

        Returns:
            `dict`: Dictionary representation of the agent.
        """
        agent_dict = super().to_dict()
        agent_dict["authorized_imports"] = self.authorized_imports
        agent_dict["executor_type"] = self.executor_type
        agent_dict["executor_kwargs"] = self.executor_kwargs
        agent_dict["max_print_outputs_length"] = self.max_print_outputs_length
        return agent_dict

    @classmethod
    def from_dict(cls, agent_dict: dict[str, Any], **kwargs) -> "CodeAgent":
        """Create CodeAgent from a dictionary representation.

        Args:
            agent_dict (`dict[str, Any]`): Dictionary representation of the agent.
            **kwargs: Additional keyword arguments that will override agent_dict values.

        Returns:
            `CodeAgent`: Instance of the CodeAgent class.
        """
        # Add CodeAgent-specific parameters to kwargs
        code_agent_kwargs = {
            "additional_authorized_imports": agent_dict.get("authorized_imports"),
            "executor_type": agent_dict.get("executor_type"),
            "executor_kwargs": agent_dict.get("executor_kwargs"),
            "max_print_outputs_length": agent_dict.get("max_print_outputs_length"),
        }
        # Filter out None values
        code_agent_kwargs = {k: v for k, v in code_agent_kwargs.items() if v is not None}
        # Update with any additional kwargs
        code_agent_kwargs.update(kwargs)
        # Call the parent class's from_dict method
        return super().from_dict(agent_dict, **code_agent_kwargs)
