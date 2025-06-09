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
监控模块 - smolagents 的监控和日志系统

本模块提供了智能代理运行过程中的监控、日志记录和性能统计功能。
主要组件包括：
- TokenUsage: 令牌使用统计
- Timing: 时间统计
- Monitor: 性能监控器
- AgentLogger: 智能日志记录器
- LogLevel: 日志级别控制

功能特性：
- 实时监控代理执行状态
- 详细的令牌消耗统计
- 精确的时间性能分析
- 丰富的可视化日志输出
- 多级别日志控制

作者: HuggingFace 团队
版本: 1.0
"""

import json
from dataclasses import dataclass, field
from enum import IntEnum

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from smolagents.utils import escape_code_brackets


__all__ = ["AgentLogger", "LogLevel", "Monitor", "TokenUsage", "Timing"]


@dataclass
class TokenUsage:
    """
    令牌使用统计类 - 记录和统计 LLM 模型的令牌消耗
    
    该类用于跟踪智能代理在执行过程中的令牌使用情况，
    包括输入令牌、输出令牌和总令牌数的统计。
    
    属性:
        input_tokens (int): 输入令牌数量，模型处理的输入文本消耗的令牌数
        output_tokens (int): 输出令牌数量，模型生成的输出文本消耗的令牌数
        total_tokens (int): 总令牌数量，自动计算的输入和输出令牌总和
    
    使用示例:
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        print(f"总令牌数: {usage.total_tokens}")  # 输出: 总令牌数: 150
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int = field(init=False)

    def __post_init__(self):
        """
        后初始化方法 - 自动计算总令牌数
        
        在对象创建后自动执行，计算输入和输出令牌的总和。
        """
        self.total_tokens = self.input_tokens + self.output_tokens

    def dict(self):
        """
        转换为字典格式
        
        将令牌使用统计信息转换为字典格式，便于序列化和存储。
        
        返回:
            dict: 包含所有令牌统计信息的字典
        """
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class Timing:
    """
    时间统计类 - 记录和计算执行时间
    
    该类用于精确记录智能代理执行步骤的时间信息，
    包括开始时间、结束时间和持续时间的计算。
    
    属性:
        start_time (float): 开始时间戳（Unix 时间戳）
        end_time (float | None): 结束时间戳，可以为 None（表示尚未结束）
    
    使用示例:
        import time
        timing = Timing(start_time=time.time())
        # ... 执行一些操作 ...
        timing.end_time = time.time()
        print(f"执行耗时: {timing.duration:.2f} 秒")
    """

    start_time: float
    end_time: float | None = None

    @property
    def duration(self):
        """
        计算持续时间
        
        根据开始时间和结束时间计算执行持续时间。
        如果结束时间为 None，则返回 None。
        
        返回:
            float | None: 持续时间（秒），如果尚未结束则返回 None
        """
        return None if self.end_time is None else self.end_time - self.start_time

    def dict(self):
        """
        转换为字典格式
        
        将时间统计信息转换为字典格式，便于序列化和存储。
        
        返回:
            dict: 包含所有时间统计信息的字典
        """
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
        }

    def __repr__(self) -> str:
        """
        字符串表示方法
        
        提供清晰的时间对象字符串表示，便于调试和日志输出。
        
        返回:
            str: 时间对象的字符串表示
        """
        return f"Timing(start_time={self.start_time}, end_time={self.end_time}, duration={self.duration})"


class Monitor:
    """
    性能监控器 - 跟踪智能代理的执行性能和资源使用
    
    该类负责监控智能代理的运行状态，收集和统计性能指标，
    包括步骤执行时间、令牌消耗统计等关键性能数据。
    
    主要功能:
    - 跟踪每个执行步骤的时间消耗
    - 统计总的令牌使用情况
    - 提供性能数据的汇总和分析
    - 支持监控数据的重置和更新
    
    参数:
        tracked_model: 被跟踪的模型对象
        logger: 日志记录器实例
    """
    
    def __init__(self, tracked_model, logger):
        """
        初始化监控器
        
        设置监控器的初始状态，准备开始收集性能数据。
        
        参数:
            tracked_model: 要监控的模型对象
            logger: 用于输出监控信息的日志记录器
        """
        self.step_durations = []  # 存储每个步骤的执行时间
        self.tracked_model = tracked_model  # 被监控的模型
        self.logger = logger  # 日志记录器
        self.total_input_token_count = 0  # 总输入令牌计数
        self.total_output_token_count = 0  # 总输出令牌计数

    def get_total_token_counts(self) -> TokenUsage:
        """
        获取总令牌使用统计
        
        返回从监控开始到当前时刻的累计令牌使用情况。
        
        返回:
            TokenUsage: 包含总输入、输出和总令牌数的统计对象
        """
        return TokenUsage(
            input_tokens=self.total_input_token_count,
            output_tokens=self.total_output_token_count,
        )

    def reset(self):
        """
        重置监控数据
        
        清空所有累积的监控数据，重新开始统计。
        通常在新的代理运行开始时调用。
        """
        self.step_durations = []
        self.total_input_token_count = 0
        self.total_output_token_count = 0

    def update_metrics(self, step_log):
        """
        更新监控指标
        
        处理新的步骤日志，更新相关的性能指标和统计数据。
        这是监控器的核心方法，每个步骤执行完成后都会调用。

        参数:
            step_log (MemoryStep): 步骤日志对象，包含执行时间和令牌使用信息
        """
        # 记录步骤执行时间
        step_duration = step_log.timing.duration
        self.step_durations.append(step_duration)
        
        # 构建控制台输出信息
        console_outputs = f"[Step {len(self.step_durations)}: Duration {step_duration:.2f} seconds"

        # 更新令牌使用统计
        if step_log.token_usage is not None:
            self.total_input_token_count += step_log.token_usage.input_tokens
            self.total_output_token_count += step_log.token_usage.output_tokens
            console_outputs += (
                f"| Input tokens: {self.total_input_token_count:,} | Output tokens: {self.total_output_token_count:,}"
            )
        console_outputs += "]"
        
        # 输出监控信息到日志
        self.logger.log(Text(console_outputs, style="dim"), level=1)


class LogLevel(IntEnum):
    """
    日志级别枚举 - 定义不同的日志输出级别
    
    该枚举类定义了智能代理系统中使用的各种日志级别，
    用于控制日志输出的详细程度和过滤不同重要性的信息。
    
    级别说明:
        OFF (-1): 关闭所有日志输出
        ERROR (0): 仅输出错误信息
        INFO (1): 输出正常信息（默认级别）
        DEBUG (2): 输出详细调试信息
    
    使用示例:
        logger = AgentLogger(level=LogLevel.DEBUG)
        logger.log("调试信息", level=LogLevel.DEBUG)
    """
    OFF = -1    # 无输出
    ERROR = 0   # 仅错误
    INFO = 1    # 正常输出（默认）
    DEBUG = 2   # 详细输出


# 定义主题颜色常量
YELLOW_HEX = "#d4b702"  # 黄色主题色，用于突出显示重要信息


class AgentLogger:
    """
    智能代理日志记录器 - 提供丰富的可视化日志输出
    
    该类是智能代理系统的核心日志组件，提供了多种格式化的日志输出方式，
    包括普通文本、Markdown、代码块、表格、树形结构等，
    使代理的运行过程更加直观和易于理解。
    
    主要特性:
    - 多级别日志控制
    - 丰富的视觉效果
    - 代码语法高亮
    - 结构化信息展示
    - 自定义样式支持
    
    参数:
        level (LogLevel): 日志输出级别，默认为 INFO
        console (Console | None): Rich 控制台对象，如果为 None 则创建新实例
    """
    
    def __init__(self, level: LogLevel = LogLevel.INFO, console: Console | None = None):
        """
        初始化日志记录器
        
        设置日志级别和输出控制台，准备开始记录日志。
        
        参数:
            level (LogLevel): 日志级别，控制输出的详细程度
            console (Console | None): Rich 控制台对象，用于格式化输出
        """
        self.level = level
        if console is None:
            self.console = Console()
        else:
            self.console = console

    def log(self, *args, level: int | str | LogLevel = LogLevel.INFO, **kwargs) -> None:
        """
        通用日志输出方法
        
        根据设定的日志级别决定是否输出消息，支持 Rich 库的所有格式化功能。
        这是所有其他日志方法的基础。

        参数:
            *args: 要输出的内容，支持多个参数
            level (int | str | LogLevel): 当前消息的日志级别
            **kwargs: 传递给 Rich Console.print 的额外参数
        """
        # 处理字符串形式的日志级别
        if isinstance(level, str):
            level = LogLevel[level.upper()]
        
        # 只有当消息级别不高于设定级别时才输出
        if level <= self.level:
            self.console.print(*args, **kwargs)

    def log_error(self, error_message: str) -> None:
        """
        输出错误消息
        
        以红色粗体格式输出错误信息，并转义代码中的特殊字符。
        
        参数:
            error_message (str): 要输出的错误消息
        """
        self.log(escape_code_brackets(error_message), style="bold red", level=LogLevel.ERROR)

    def log_markdown(self, content: str, title: str | None = None, level=LogLevel.INFO, style=YELLOW_HEX) -> None:
        """
        输出 Markdown 格式的内容
        
        将内容以 Markdown 格式进行语法高亮显示，可选择添加标题。
        适用于显示格式化的文档、说明或结构化信息。
        
        参数:
            content (str): Markdown 格式的内容
            title (str | None): 可选的标题
            level (LogLevel): 日志级别
            style (str): 标题的样式
        """
        # 创建 Markdown 语法高亮对象
        markdown_content = Syntax(
            content,
            lexer="markdown",
            theme="github-dark",
            word_wrap=True,
        )
        
        # 如果有标题，则添加标题规则
        if title:
            self.log(
                Group(
                    Rule(
                        "[bold italic]" + title,
                        align="left",
                        style=style,
                    ),
                    markdown_content,
                ),
                level=level,
            )
        else:
            self.log(markdown_content, level=level)

    def log_code(self, title: str, content: str, level: int = LogLevel.INFO) -> None:
        """
        输出代码块
        
        以带有语法高亮的面板形式显示代码，适用于展示执行的代码片段。
        
        参数:
            title (str): 代码块的标题
            content (str): 要显示的代码内容
            level (int): 日志级别
        """
        self.log(
            Panel(
                Syntax(
                    content,
                    lexer="python",  # 使用 Python 语法高亮
                    theme="monokai",  # 使用 Monokai 主题
                    word_wrap=True,
                ),
                title="[bold]" + title,
                title_align="left",
                box=box.HORIZONTALS,  # 使用水平线框样式
            ),
            level=level,
        )

    def log_rule(self, title: str, level: int = LogLevel.INFO) -> None:
        """
        输出分隔线规则
        
        显示一条带有标题的分隔线，用于分隔不同的执行步骤或阶段。
        
        参数:
            title (str): 分隔线上的标题文本
            level (int): 日志级别
        """
        self.log(
            Rule(
                "[bold]" + title,
                characters="━",
                style=YELLOW_HEX,
            ),
            level=LogLevel.INFO,
        )

    def log_task(self, content: str, subtitle: str, title: str | None = None, level: LogLevel = LogLevel.INFO) -> None:
        """
        输出任务信息
        
        以突出的面板形式显示新任务的开始，包括任务内容和相关信息。
        
        参数:
            content (str): 任务的具体内容
            subtitle (str): 子标题，通常包含模型信息
            title (str | None): 可选的主标题
            level (LogLevel): 日志级别
        """
        self.log(
            Panel(
                f"\n[bold]{escape_code_brackets(content)}\n",
                title="[bold]New run" + (f" - {title}" if title else ""),
                subtitle=subtitle,
                border_style=YELLOW_HEX,  # 使用黄色边框
                subtitle_align="left",
            ),
            level=level,
        )

    def log_messages(self, messages: list[dict], level: LogLevel = LogLevel.DEBUG) -> None:
        """
        输出消息列表
        
        以格式化的 JSON 形式显示消息列表，主要用于调试目的。
        
        参数:
            messages (list[dict]): 要显示的消息列表
            level (LogLevel): 日志级别，默认为 DEBUG
        """
        # 将消息列表转换为格式化的 JSON 字符串
        messages_as_string = "\n".join([json.dumps(dict(message), indent=4) for message in messages])
        self.log(
            Syntax(
                messages_as_string,
                lexer="markdown",
                theme="github-dark",
                word_wrap=True,
            ),
            level=level,
        )

    def visualize_agent_tree(self, agent):
        """
        可视化智能代理的结构树
        
        以树形结构展示智能代理的完整架构，包括工具、子代理等组件。
        这是一个复杂的可视化方法，帮助用户理解代理的内部结构。
        
        参数:
            agent: 要可视化的智能代理对象
        """
        
        def create_tools_section(tools_dict):
            """
            创建工具信息表格
            
            将代理的工具信息以表格形式组织，显示工具名称、描述和参数。
            
            参数:
                tools_dict (dict): 工具字典
                
            返回:
                Group: 包含标题和表格的组合对象
            """
            table = Table(show_header=True, header_style="bold")
            table.add_column("Name", style="#1E90FF")  # 蓝色工具名称
            table.add_column("Description")
            table.add_column("Arguments")

            # 遍历工具字典，添加每个工具的信息
            for name, tool in tools_dict.items():
                # 格式化工具参数信息
                args = [
                    f"{arg_name} (`{info.get('type', 'Any')}`{', optional' if info.get('optional') else ''}): {info.get('description', '')}"
                    for arg_name, info in getattr(tool, "inputs", {}).items()
                ]
                table.add_row(name, getattr(tool, "description", str(tool)), "\n".join(args))

            return Group("🛠️ [italic #1E90FF]Tools:[/italic #1E90FF]", table)

        def get_agent_headline(agent, name: str | None = None):
            """
            生成代理标题行
            
            创建包含代理信息的标题字符串。
            
            参数:
                agent: 代理对象
                name (str | None): 可选的代理名称
                
            返回:
                str: 格式化的代理标题
            """
            name_headline = f"{name} | " if name else ""
            return f"[bold {YELLOW_HEX}]{name_headline}{agent.__class__.__name__} | {agent.model.model_id}"

        def build_agent_tree(parent_tree, agent_obj):
            """
            递归构建代理树结构
            
            这是一个递归函数，用于构建代理及其子代理的完整树形结构。
            
            参数:
                parent_tree: 父级树节点
                agent_obj: 要处理的代理对象
            """
            # 添加工具信息
            parent_tree.add(create_tools_section(agent_obj.tools))

            # 如果有管理的子代理，递归添加它们
            if agent_obj.managed_agents:
                agents_branch = parent_tree.add("🤖 [italic #1E90FF]Managed agents:")
                for name, managed_agent in agent_obj.managed_agents.items():
                    agent_tree = agents_branch.add(get_agent_headline(managed_agent, name))
                    
                    # 为 CodeAgent 添加特殊信息
                    if managed_agent.__class__.__name__ == "CodeAgent":
                        agent_tree.add(
                            f"✅ [italic #1E90FF]Authorized imports:[/italic #1E90FF] {managed_agent.additional_authorized_imports}"
                        )
                    
                    # 添加代理描述
                    agent_tree.add(f"📝 [italic #1E90FF]Description:[/italic #1E90FF] {managed_agent.description}")
                    
                    # 递归构建子代理树
                    build_agent_tree(agent_tree, managed_agent)

        # 创建主树结构
        main_tree = Tree(get_agent_headline(agent))
        
        # 为 CodeAgent 添加特殊信息
        if agent.__class__.__name__ == "CodeAgent":
            main_tree.add(
                f"✅ [italic #1E90FF]Authorized imports:[/italic #1E90FF] {agent.additional_authorized_imports}"
            )
        
        # 构建完整的代理树
        build_agent_tree(main_tree, agent)
        
        # 输出树形结构
        self.console.print(main_tree)
