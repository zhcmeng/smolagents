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
模型模块 - smolagents 的多模型支持系统

本模块提供了多种 LLM 模型的统一接口，支持：
- 本地模型（Transformers、vLLM、MLX）
- 云端 API 模型（OpenAI、HuggingFace、LiteLLM）
- 工具调用和结构化输出
- 流式生成和批量处理

主要类：
- Model: 模型的抽象基类
- TransformersModel: 基于 Transformers 的本地模型
- OpenAIServerModel: OpenAI 兼容的 API 模型
- InferenceClientModel: HuggingFace Inference API 模型
- VLLMModel: 基于 vLLM 的高性能模型

支持的提供商：
- OpenAI、Azure OpenAI
- HuggingFace Hub
- Amazon Bedrock
- 自定义 OpenAI 兼容服务器

作者: HuggingFace 团队
版本: 1.0
"""

import json
import logging
import os
import re
import uuid
import warnings
from collections.abc import Generator
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
from threading import Thread
from typing import TYPE_CHECKING, Any

from .monitoring import TokenUsage
from .tools import Tool
from .utils import _is_package_available, encode_image_base64, make_image_url, parse_json_blob


if TYPE_CHECKING:
    from transformers import StoppingCriteriaList


logger = logging.getLogger(__name__)

STRUCTURED_GENERATION_PROVIDERS = ["cerebras", "fireworks-ai"]
CODEAGENT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "schema": {
            "additionalProperties": False,
            "properties": {
                "thought": {
                    "description": "A free form text description of the thought process.",
                    "title": "Thought",
                    "type": "string",
                },
                "code": {
                    "description": "Valid Python code snippet implementing the thought.",
                    "title": "Code",
                    "type": "string",
                },
            },
            "required": ["thought", "code"],
            "title": "ThoughtAndCodeAnswer",
            "type": "object",
        },
        "name": "ThoughtAndCodeAnswer",
        "strict": True,
    },
}


def get_dict_from_nested_dataclasses(obj, ignore_key=None):
    """
    将嵌套的数据类对象转换为字典
    
    参数:
        obj: 要转换的对象
        ignore_key: 要忽略的键名
        
    返回:
        dict: 转换后的字典
    """
    def convert(obj):
        """
        递归转换函数
        
        参数:
            obj: 要转换的对象
            
        返回:
            转换后的对象或字典
        """
        if hasattr(obj, "__dataclass_fields__"):
            return {k: convert(v) for k, v in asdict(obj).items() if k != ignore_key}
        return obj

    return convert(obj)


@dataclass
class ChatMessageToolCallDefinition:
    """
    聊天消息工具调用定义
    
    定义工具调用的具体参数和元数据，包含工具名称、参数和描述信息。
    用于结构化地表示 LLM 调用工具时的参数信息。
    
    属性:
        arguments (Any): 工具调用的参数，可以是字典、字符串或其他类型
        name (str): 工具的名称标识符
        description (str | None): 工具的描述信息，可选
    """
    arguments: Any
    name: str
    description: str | None = None


@dataclass
class ChatMessageToolCall:
    """
    聊天消息工具调用对象
    
    表示一个完整的工具调用实例，包含工具调用的所有必要信息：
    函数定义、调用ID和调用类型。通常由 LLM 生成，用于执行特定的工具操作。
    
    属性:
        function (ChatMessageToolCallDefinition): 工具调用的函数定义
        id (str): 工具调用的唯一标识符
        type (str): 工具调用的类型，通常为 "function"
    """
    function: ChatMessageToolCallDefinition
    id: str
    type: str

    def __str__(self) -> str:
        """
        返回工具调用的字符串表示
        
        生成一个易读的字符串，显示工具调用的关键信息，
        包括调用ID、工具名称和参数。用于调试和日志记录。
        
        返回:
            str: 格式化的工具调用字符串，包含ID、工具名和参数
        """
        return f"Call: {self.id}: Calling {str(self.function.name)} with arguments: {str(self.function.arguments)}"


@dataclass
class ChatMessage:
    """
    聊天消息数据类
    
    表示聊天对话中的一条消息，支持文本内容和工具调用。
    这是 smolagents 系统中消息传递的核心数据结构，用于：
    - 存储用户输入、系统提示和 LLM 响应
    - 记录工具调用信息和执行结果
    - 跟踪令牌使用情况和原始API响应
    
    属性:
        role (str): 消息角色，如 "user"、"assistant"、"system"
        content (str | None): 消息的文本内容，可为空（如纯工具调用）
        tool_calls (list[ChatMessageToolCall] | None): 工具调用列表，可选
        raw (Any | None): 存储API的原始响应数据，用于调试和扩展
        token_usage (TokenUsage | None): 令牌使用统计信息，可选
    """
    role: str
    content: str | None = None
    tool_calls: list[ChatMessageToolCall] | None = None
    raw: Any | None = None  # Stores the raw output from the API
    token_usage: TokenUsage | None = None

    def model_dump_json(self):
        """
        将聊天消息对象转换为JSON字符串
        
        序列化消息对象为JSON格式，自动排除敏感的raw字段。
        适用于消息存储、传输和日志记录等场景。
        
        返回:
            str: 序列化后的JSON字符串（排除raw字段以减少数据量）
        """
        return json.dumps(get_dict_from_nested_dataclasses(self, ignore_key="raw"))

    @classmethod
    def from_dict(cls, data: dict, raw: Any | None = None, token_usage: TokenUsage | None = None) -> "ChatMessage":
        """
        从字典创建ChatMessage对象
        
        反序列化字典数据为ChatMessage对象，自动处理嵌套的工具调用结构。
        常用于从JSON数据、API响应或存储中恢复消息对象。
        
        参数:
            data (dict): 包含消息数据的字典，必须包含 "role" 字段
            raw (Any, 可选): 原始API响应数据，用于保留完整的响应信息
            token_usage (TokenUsage, 可选): 令牌使用统计信息
            
        返回:
            ChatMessage: 根据字典数据创建的聊天消息对象
        """
        if data.get("tool_calls"):
            tool_calls = [
                ChatMessageToolCall(
                    function=ChatMessageToolCallDefinition(**tc["function"]), id=tc["id"], type=tc["type"]
                )
                for tc in data["tool_calls"]
            ]
            data["tool_calls"] = tool_calls
        return cls(
            role=data["role"],
            content=data.get("content"),
            tool_calls=data.get("tool_calls"),
            raw=raw,
            token_usage=token_usage,
        )

    def dict(self):
        """
        将ChatMessage对象转换为字典
        
        将消息对象序列化为字典格式，保留所有字段信息。
        适用于数据传输、存储和与外部系统的集成。
        
        返回:
            dict: 包含所有消息字段的字典表示（排除raw字段）
        """
        return get_dict_from_nested_dataclasses(self)


def parse_json_if_needed(arguments: str | dict) -> str | dict:
    """
    如果需要，将字符串参数解析为JSON对象
    
    参数:
        arguments (str | dict): 要解析的参数，可能是字符串或已经是字典
        
    返回:
        str | dict: 如果是有效JSON则返回解析后的字典，否则返回原始参数
    """
    if isinstance(arguments, dict):
        return arguments
    else:
        try:
            return json.loads(arguments)
        except Exception:
            return arguments


@dataclass
class ChatMessageStreamDelta:
    """
    聊天消息流式增量数据
    
    表示流式生成过程中的一个增量更新，用于实时传输 LLM 生成的内容。
    支持渐进式内容更新和工具调用信息的流式传输。
    
    流式生成的优势:
    - 降低用户感知延迟
    - 提供实时反馈
    - 支持长文本的渐进式显示
    - 改善用户体验
    
    属性:
        content (str | None): 本次增量的文本内容，可为空
        tool_calls (list[ChatMessageToolCall] | None): 本次增量的工具调用信息，可选
        token_usage (TokenUsage | None): 令牌使用统计（通常在流式结束时提供）
    """
    content: str | None = None
    tool_calls: list[ChatMessageToolCall] | None = None
    token_usage: TokenUsage | None = None


@dataclass
class ToolCallStreamDelta:
    """
    工具调用流式增量数据
    
    表示在流式生成过程中工具调用的增量更新。
    由于工具调用信息可能分多次传输，此类用于累积构建完整的工具调用。
    
    流式工具调用的特点:
    - 工具名称可能先于参数到达
    - 参数可能分段传输
    - 需要根据索引正确组装多个工具调用
    
    属性:
        index (int | None): 工具调用在列表中的索引位置
        id (str | None): 工具调用的唯一标识符
        type (str | None): 工具调用类型，通常为 "function"
        function (dict[str, Any] | None): 函数调用信息，包含名称和参数增量
    """

    index: int | None = None
    id: str | None = None
    type: str | None = None
    function: dict[str, Any] | None = None


class MessageRole(str, Enum):
    """
    消息角色枚举
    
    定义聊天对话中可能出现的所有消息角色类型。
    这些角色用于区分消息的来源和性质，帮助 LLM 理解对话上下文。
    
    角色说明:
    - USER: 用户输入的消息，包含任务、问题或指令
    - ASSISTANT: LLM 助手的响应消息，包含分析、回答或工具调用
    - SYSTEM: 系统提示消息，定义助手的身份和行为规范
    - TOOL_CALL: 工具调用消息，表示执行特定功能的请求
    - TOOL_RESPONSE: 工具响应消息，包含工具执行的结果
    
    用途:
    - 消息分类和路由
    - 上下文理解和维护
    - 权限控制和安全检查
    - 对话流程管理
    """
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_CALL = "tool-call"
    TOOL_RESPONSE = "tool-response"

    @classmethod
    def roles(cls):
        """
        获取所有可用的消息角色
        
        返回系统支持的所有消息角色的字符串值列表。
        常用于验证角色有效性和生成角色选择列表。
        
        返回:
            list[str]: 所有角色的字符串值列表，按定义顺序排列
        """
        return [r.value for r in cls]


tool_role_conversions = {
    MessageRole.TOOL_CALL: MessageRole.ASSISTANT,
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}


def get_tool_json_schema(tool: Tool) -> dict:
    """
    将Tool对象转换为OpenAI兼容的JSON Schema格式
    
    此函数将smolagents的Tool对象转换为符合OpenAI工具调用API标准的JSON模式。
    转换过程包括类型映射、必需字段识别和结构重组。
    
    转换规则:
    - "any" 类型映射为 "string" 类型
    - 非nullable字段自动标记为required
    - 保持原始的参数描述和约束
    - 生成符合OpenAI函数调用格式的嵌套结构
    
    参数:
        tool (Tool): smolagents工具对象，包含名称、描述和输入规范
        
    返回:
        dict: 符合OpenAI函数调用格式的工具模式字典，包含：
            - type: 固定为 "function"
            - function: 包含name、description和parameters的函数定义
    """
    properties = deepcopy(tool.inputs)
    required = []
    for key, value in properties.items():
        if value["type"] == "any":
            value["type"] = "string"
        if not ("nullable" in value and value["nullable"]):
            required.append(key)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def remove_stop_sequences(content: str, stop_sequences: list[str]) -> str:
    """
    从内容末尾移除停止序列
    
    参数:
        content (str): 要处理的文本内容
        stop_sequences (list[str]): 停止序列列表
        
    返回:
        str: 移除停止序列后的内容
    """
    for stop_seq in stop_sequences:
        if content[-len(stop_seq) :] == stop_seq:
            content = content[: -len(stop_seq)]
    return content


def get_clean_message_list(
    message_list: list[dict[str, str | list[dict]]],
    role_conversions: dict[MessageRole, MessageRole] | dict[str, str] = {},
    convert_images_to_image_urls: bool = False,
    flatten_messages_as_text: bool = False,
) -> list[dict[str, str | list[dict]]]:
    """
    清理和标准化消息列表
    
    对原始消息列表进行预处理，确保与不同LLM的聊天模板兼容。
    主要处理包括角色转换、消息合并、图像编码和格式标准化。
    
    处理功能:
    1. 角色转换：将特殊角色映射为标准角色
    2. 消息合并：合并相同角色的连续消息
    3. 图像处理：编码图像为base64或转换为URL格式
    4. 文本扁平化：将复杂消息结构简化为纯文本
    5. 格式验证：确保所有消息角色有效
    
    用途:
    - 适配不同LLM提供商的API格式
    - 减少消息数量以节省token
    - 标准化图像处理流程
    - 兼容各种聊天模板
    
    参数:
        message_list (list[dict]): 原始聊天消息列表
        role_conversions (dict, 可选): 角色转换映射，如 {"tool-call": "assistant"}
        convert_images_to_image_urls (bool, 默认False): 是否将图像转换为image_url格式
        flatten_messages_as_text (bool, 默认False): 是否将消息内容扁平化为纯文本
        
    返回:
        list[dict]: 清理和标准化后的消息列表，兼容transformers聊天模板
        
    异常:
        ValueError: 当消息包含无效角色时抛出
    """
    output_message_list: list[dict[str, str | list[dict]]] = []
    message_list = deepcopy(message_list)  # Avoid modifying the original list
    for message in message_list:
        role = message["role"]
        if role not in MessageRole.roles():
            raise ValueError(f"Incorrect role {role}, only {MessageRole.roles()} are supported for now.")

        if role in role_conversions:
            message["role"] = role_conversions[role]  # type: ignore
        # encode images if needed
        if isinstance(message["content"], list):
            for element in message["content"]:
                assert isinstance(element, dict), "Error: this element should be a dict:" + str(element)
                if element["type"] == "image":
                    assert not flatten_messages_as_text, f"Cannot use images with {flatten_messages_as_text=}"
                    if convert_images_to_image_urls:
                        element.update(
                            {
                                "type": "image_url",
                                "image_url": {"url": make_image_url(encode_image_base64(element.pop("image")))},
                            }
                        )
                    else:
                        element["image"] = encode_image_base64(element["image"])

        if len(output_message_list) > 0 and message["role"] == output_message_list[-1]["role"]:
            assert isinstance(message["content"], list), "Error: wrong content:" + str(message["content"])
            if flatten_messages_as_text:
                output_message_list[-1]["content"] += "\n" + message["content"][0]["text"]
            else:
                for el in message["content"]:
                    if el["type"] == "text" and output_message_list[-1]["content"][-1]["type"] == "text":
                        # Merge consecutive text messages rather than creating new ones
                        output_message_list[-1]["content"][-1]["text"] += "\n" + el["text"]
                    else:
                        output_message_list[-1]["content"].append(el)
        else:
            if flatten_messages_as_text:
                content = message["content"][0]["text"]
            else:
                content = message["content"]
            output_message_list.append({"role": message["role"], "content": content})
    return output_message_list


def get_tool_call_from_text(text: str, tool_name_key: str, tool_arguments_key: str) -> ChatMessageToolCall:
    """
    从文本中解析工具调用信息
    
    解析LLM生成的文本，提取其中的工具调用信息并构造标准的工具调用对象。
    此函数处理不支持原生工具调用的LLM，通过文本解析实现工具调用功能。
    
    解析流程:
    1. 从文本中提取JSON格式的工具调用信息
    2. 根据指定键名提取工具名称和参数
    3. 处理参数的类型转换（字符串转JSON等）
    4. 生成唯一的调用ID
    5. 构造完整的工具调用对象
    
    适用场景:
    - 不支持原生工具调用的LLM模型
    - 需要从自由文本中提取结构化工具调用
    - 自定义工具调用格式的解析
    
    参数:
        text (str): 包含工具调用JSON信息的文本
        tool_name_key (str): JSON中工具名称字段的键名，如 "name"
        tool_arguments_key (str): JSON中工具参数字段的键名，如 "arguments"
        
    返回:
        ChatMessageToolCall: 解析并构造的工具调用对象，包含随机生成的ID
        
    异常:
        ValueError: 当找不到指定的工具名称键或JSON解析失败时抛出
    """
    tool_call_dictionary, _ = parse_json_blob(text)
    try:
        tool_name = tool_call_dictionary[tool_name_key]
    except Exception as e:
        raise ValueError(
            f"Key {tool_name_key=} not found in the generated tool call. Got keys: {list(tool_call_dictionary.keys())} instead"
        ) from e
    tool_arguments = tool_call_dictionary.get(tool_arguments_key, None)
    if isinstance(tool_arguments, str):
        tool_arguments = parse_json_if_needed(tool_arguments)
    return ChatMessageToolCall(
        id=str(uuid.uuid4()),
        type="function",
        function=ChatMessageToolCallDefinition(name=tool_name, arguments=tool_arguments),
    )


def supports_stop_parameter(model_id: str) -> bool:
    """
    检查模型是否支持stop参数
    
    某些OpenAI推理模型（如o3和o4-mini系列）不支持stop参数。
    此函数通过模型名称模式匹配来判断是否支持stop功能。
    
    不支持stop的模型:
    - o3 系列：o3, o3-2025-04-16 等版本变体
    - o4-mini 系列：o4-mini, o4-mini-2025-04-16 等版本变体
    
    用途:
    - API调用前的兼容性检查
    - 动态调整生成参数
    - 避免不支持的参数导致的API错误
    
    参数:
        model_id (str): 完整的模型标识符，如 "openai/o3", "o4-mini-2025-04-16"
        
    返回:
        bool: 如果模型支持stop参数则返回True，否则返回False
        
    示例:
        >>> supports_stop_parameter("openai/gpt-4")
        True
        >>> supports_stop_parameter("openai/o3")
        False
        >>> supports_stop_parameter("o4-mini-2025-04-16")
        False
    """
    model_name = model_id.split("/")[-1]
    # o3 and o4-mini (including versioned variants, o3-2025-04-16) don't support stop parameter
    pattern = r"^(o3[-\d]*|o4-mini[-\d]*)$"
    return not re.match(pattern, model_name)


class Model:
    """
    模型抽象基类 - 为所有 LLM 模型提供统一接口
    
    该类定义了与 LLM 交互的标准接口，所有具体的模型实现都应继承此类。
    
    主要功能:
    - 统一的消息处理和生成接口
    - 工具调用支持
    - 结构化输出格式
    - 令牌使用统计
    - 停止序列处理
    
    参数:
        flatten_messages_as_text (bool, 默认 False): 是否将消息扁平化为纯文本
        tool_name_key (str, 默认 "name"): 工具调用中工具名称的键名
        tool_arguments_key (str, 默认 "arguments"): 工具调用中参数的键名
        model_id (str, 可选): 模型标识符
        **kwargs: 传递给具体模型实现的其他参数
    """
    
    def __init__(
        self,
        flatten_messages_as_text: bool = False,
        tool_name_key: str = "name",
        tool_arguments_key: str = "arguments",
        model_id: str | None = None,
        **kwargs,
    ):
        self.flatten_messages_as_text = flatten_messages_as_text
        self.tool_name_key = tool_name_key
        self.tool_arguments_key = tool_arguments_key
        self.kwargs = kwargs
        self._last_input_token_count: int | None = None
        self._last_output_token_count: int | None = None
        self.model_id: str | None = model_id

    @property
    def last_input_token_count(self) -> int | None:
        warnings.warn(
            "Attribute last_input_token_count is deprecated and will be removed in version 1.20. "
            "Please use TokenUsage.input_tokens instead.",
            FutureWarning,
        )
        return self._last_input_token_count

    @property
    def last_output_token_count(self) -> int | None:
        warnings.warn(
            "Attribute last_output_token_count is deprecated and will be removed in version 1.20. "
            "Please use TokenUsage.output_tokens instead.",
            FutureWarning,
        )
        return self._last_output_token_count

    def _prepare_completion_kwargs(
        self,
        messages: list[dict[str, str | list[dict]]],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        convert_images_to_image_urls: bool = False,
        tool_choice: str | dict | None = "required",  # Configurable tool_choice parameter
        **kwargs,
    ) -> dict[str, Any]:
        """
        准备模型调用所需的参数，处理参数优先级和格式转换
        
        这是所有模型子类的核心参数准备方法，负责：
        1. 消息格式标准化和清理
        2. 参数优先级处理和合并
        3. 工具调用配置生成
        4. 特殊参数的条件处理
        5. 兼容性检查和调整
        
        参数优先级（从高到低）：
        1. 显式传递的 kwargs 参数（最高优先级）
        2. 方法特定参数（stop_sequences、response_format等）
        3. self.kwargs 中的默认值（最低优先级）
        
        参数:
            messages (list[dict]): 原始消息列表，将被清理和标准化
            stop_sequences (list[str], 可选): 停止序列，某些模型可能不支持
            response_format (dict, 可选): 结构化输出格式定义
            tools_to_call_from (list[Tool], 可选): 可用工具列表，将转换为JSON模式
            custom_role_conversions (dict, 可选): 自定义角色转换映射
            convert_images_to_image_urls (bool): 是否转换图像为URL格式
            tool_choice (str | dict | None): 工具选择策略，默认为"required"
            **kwargs: 其他模型特定参数，具有最高优先级
            
        返回:
            dict[str, Any]: 准备好的完整参数字典，可直接用于模型调用
        """
        # 从kwargs中提取flatten_messages_as_text设置
        flatten_messages_as_text = kwargs.pop("flatten_messages_as_text", self.flatten_messages_as_text)

        # 清理和标准化消息列表
        messages = get_clean_message_list(
            messages,
            role_conversions=custom_role_conversions or tool_role_conversions,
            convert_images_to_image_urls=convert_images_to_image_urls,
            flatten_messages_as_text=flatten_messages_as_text,
        )

        # 使用 self.kwargs 作为基础配置
        completion_kwargs = {
            **self.kwargs,
            "messages": messages,
        }

        # 处理特定参数
        if stop_sequences is not None:
            # 某些模型不支持stop参数
            if supports_stop_parameter(self.model_id or ""):
                completion_kwargs["stop"] = stop_sequences
        if response_format is not None:
            completion_kwargs["response_format"] = response_format

        # 处理工具参数
        if tools_to_call_from:
            tools_config = {
                "tools": [get_tool_json_schema(tool) for tool in tools_to_call_from],
            }
            if tool_choice is not None:
                tools_config["tool_choice"] = tool_choice
            completion_kwargs.update(tools_config)

        # 最后，用传入的kwargs覆盖所有设置
        completion_kwargs.update(kwargs)

        return completion_kwargs

    def generate(
        self,
        messages: list[dict[str, str | list[dict]] | ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        """
        处理输入消息并返回模型响应
        
        参数:
            messages (list[dict] | list[ChatMessage]): 要处理的消息列表。
                每个字典应该有结构 {"role": "user/system", "content": "消息内容"}
            stop_sequences (list[str], 可选): 停止序列列表，
                如果在模型输出中遇到这些字符串将停止生成
            response_format (dict[str, str], 可选): 模型响应中使用的响应格式
            tools_to_call_from (list[Tool], 可选): 模型可以用来生成响应的工具列表
            **kwargs: 传递给底层模型的其他关键字参数
            
        返回:
            ChatMessage: 包含模型响应的聊天消息对象
        """
        raise NotImplementedError("This method must be implemented in child classes")

    def __call__(self, *args, **kwargs):
        """
        使模型对象可调用，直接调用generate方法
        
        参数:
            *args: 位置参数，传递给generate方法
            **kwargs: 关键字参数，传递给generate方法
            
        返回:
            ChatMessage: generate方法的返回值
        """
        return self.generate(*args, **kwargs)

    def parse_tool_calls(self, message: ChatMessage) -> ChatMessage:
        """
        解析消息中的工具调用
        
        某些API或模型不会将工具调用作为结构化对象返回，而是包含在文本内容中。
        此方法负责从消息内容中提取和解析工具调用信息，并将其标准化。
        
        解析过程:
        1. 确保消息角色设置为 ASSISTANT
        2. 检查是否已存在工具调用对象
        3. 如果没有，从消息内容中解析工具调用
        4. 标准化参数格式（JSON字符串转对象等）
        5. 验证解析结果的完整性
        
        适用场景:
        - 不支持原生工具调用的模型
        - 文本格式的工具调用响应
        - 需要后处理的工具调用信息
        
        参数:
            message (ChatMessage): 包含工具调用信息的聊天消息
            
        返回:
            ChatMessage: 解析后包含标准化工具调用对象的消息
            
        异常:
            AssertionError: 当消息既无内容又无工具调用，或解析失败时抛出
        """
        message.role = MessageRole.ASSISTANT  # Overwrite role if needed
        if not message.tool_calls:
            assert message.content is not None, "Message contains no content and no tool calls"
            message.tool_calls = [
                get_tool_call_from_text(message.content, self.tool_name_key, self.tool_arguments_key)
            ]
        assert len(message.tool_calls) > 0, "No tool call was found in the model output"
        for tool_call in message.tool_calls:
            tool_call.function.arguments = parse_json_if_needed(tool_call.function.arguments)
        return message

    def to_dict(self) -> dict:
        """
        将模型转换为JSON兼容的字典
        
        序列化模型配置为字典格式，用于保存、传输和恢复模型配置。
        自动排除敏感信息（如API密钥）以确保安全性。
        
        序列化内容:
        - 基础配置：model_id、temperature、max_tokens等
        - 网络配置：api_base、timeout、provider等
        - 硬件配置：device_map、torch_dtype等
        - 服务配置：organization、project、azure_endpoint等
        - 自定义配置：kwargs中的其他参数
        
        安全特性:
        - 自动排除敏感属性（token、api_key）
        - 提供安全警告提示用户手动处理
        - 确保序列化结果可安全共享
        
        用途:
        - 模型配置的持久化存储
        - 配置的版本控制和备份
        - 模型配置的跨环境迁移
        - 配置的可视化和调试
        
        返回:
            dict: 包含模型配置的字典，已排除敏感信息，可安全序列化
        """
        model_dictionary = {
            **self.kwargs,
            "model_id": self.model_id,
        }
        for attribute in [
            "custom_role_conversion",
            "temperature",
            "max_tokens",
            "provider",
            "timeout",
            "api_base",
            "torch_dtype",
            "device_map",
            "organization",
            "project",
            "azure_endpoint",
        ]:
            if hasattr(self, attribute):
                model_dictionary[attribute] = getattr(self, attribute)

        dangerous_attributes = ["token", "api_key"]
        for attribute_name in dangerous_attributes:
            if hasattr(self, attribute_name):
                print(
                    f"For security reasons, we do not export the `{attribute_name}` attribute of your model. Please export it manually."
                )
        return model_dictionary

    @classmethod
    def from_dict(cls, model_dictionary: dict[str, Any]) -> "Model":
        """
        从字典创建模型实例
        
        反序列化模型配置字典，恢复完整的模型实例。
        这是 to_dict 方法的逆操作，用于从序列化的配置中重建模型。
        
        恢复过程:
        1. 解析配置字典中的所有参数
        2. 将参数传递给类构造函数
        3. 自动处理类型转换和验证
        4. 返回完全配置的模型实例
        
        注意事项:
        - 敏感信息（如API密钥）需要单独设置
        - 某些运行时状态不会被恢复
        - 依赖的库和环境需要预先准备
        
        适用场景:
        - 从配置文件加载模型
        - 模型配置的热重载
        - 分布式环境中的模型同步
        - 测试和调试中的模型重建
        
        参数:
            model_dictionary (dict): 包含完整模型配置的字典
            
        返回:
            Model: 根据配置创建的模型实例，可直接使用
        """
        return cls(**{k: v for k, v in model_dictionary.items()})


class VLLMModel(Model):
    """
    vLLM 高性能推理模型
    
    使用 [vLLM](https://docs.vllm.ai/) 框架进行快速 LLM 推理和服务。
    vLLM 是一个高性能的大语言模型推理引擎，专为生产环境优化。
    
    主要特性:
    - 高吞吐量：支持大批量并发推理
    - 低延迟：优化的内存管理和调度
    - 动态批处理：自动批量处理请求
    - 高效内存使用：PagedAttention 技术
    - 多种模型支持：兼容 HuggingFace 模型
    
    性能优势:
    - 比标准 Transformers 快 2-24 倍
    - 更高的 GPU 利用率
    - 更好的并发处理能力
    - 生产级的稳定性和可靠性
    
    适用场景:
    - 高并发推理服务
    - 生产环境部署
    - 批量文本处理
    - 性能敏感的应用
    
    参数:
        model_id (str): 用于推理的 Hugging Face 模型 ID，
            可以是模型路径或 HuggingFace 模型库的标识符
        model_kwargs (dict[str, Any], 可选): 传递给 vLLM 模型的额外参数，
            如 revision、max_model_len 等
    """

    def __init__(
        self,
        model_id,
        model_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        if not _is_package_available("vllm"):
            raise ModuleNotFoundError("Please install 'vllm' extra to use VLLMModel: `pip install 'smolagents[vllm]'`")

        from vllm import LLM  # type: ignore
        from vllm.transformers_utils.tokenizer import get_tokenizer  # type: ignore

        self.model_kwargs = model_kwargs or {}
        super().__init__(**kwargs)
        self.model_id = model_id
        self.model = LLM(model=model_id, **self.model_kwargs)
        assert self.model is not None
        self.tokenizer = get_tokenizer(model_id)
        self._is_vlm = False  # VLLMModel does not support vision models yet.

    def cleanup(self):
        """
        清理vLLM模型资源
        
        释放GPU内存，销毁分布式环境，清理模型相关资源。
        在不再使用模型时调用此方法可以释放系统资源。
        """
        import gc

        import torch
        from vllm.distributed.parallel_state import (  # type: ignore
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        if self.model is not None:
            # taken from https://github.com/vllm-project/vllm/issues/1908#issuecomment-2076870351
            del self.model.llm_engine.model_executor.driver_worker
        gc.collect()
        destroy_distributed_environment()
        torch.cuda.empty_cache()

    def generate(
        self,
        messages: list[dict[str, str | list[dict]] | ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        from vllm import SamplingParams  # type: ignore

        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            flatten_messages_as_text=(not self._is_vlm),
            stop_sequences=stop_sequences,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )
        # Override the OpenAI schema for VLLM compatibility
        guided_options_request = {"guided_json": response_format["json_schema"]["schema"]} if response_format else None

        messages = completion_kwargs.pop("messages")
        prepared_stop_sequences = completion_kwargs.pop("stop", [])
        tools = completion_kwargs.pop("tools", None)
        completion_kwargs.pop("tool_choice", None)

        if tools_to_call_from is not None:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )

        sampling_params = SamplingParams(
            n=kwargs.get("n", 1),
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 2048),
            stop=prepared_stop_sequences,
        )

        out = self.model.generate(
            prompt,
            sampling_params=sampling_params,
            guided_options_request=guided_options_request,
        )

        output_text = out[0].outputs[0].text
        self._last_input_token_count = len(out[0].prompt_token_ids)
        self._last_output_token_count = len(out[0].outputs[0].token_ids)
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=output_text,
            raw={"out": output_text, "completion_kwargs": completion_kwargs},
            token_usage=TokenUsage(
                input_tokens=len(out[0].prompt_token_ids),
                output_tokens=len(out[0].outputs[0].token_ids),
            ),
        )


class MLXModel(Model):
    """A class to interact with models loaded using MLX on Apple silicon.

    > [!TIP]
    > You must have `mlx-lm` installed on your machine. Please run `pip install smolagents[mlx-lm]` if it's not the case.

    Parameters:
        model_id (str):
            The Hugging Face model ID to be used for inference. This can be a path or model identifier from the Hugging Face model hub.
        tool_name_key (str):
            The key, which can usually be found in the model's chat template, for retrieving a tool name.
        tool_arguments_key (str):
            The key, which can usually be found in the model's chat template, for retrieving tool arguments.
        trust_remote_code (bool):
            Some models on the Hub require running remote code: for this model, you would have to set this flag to True.
        kwargs (dict, *optional*):
            Any additional keyword arguments that you want to use in model.generate(), for instance `max_tokens`.

    Example:
    ```python
    >>> engine = MLXModel(
    ...     model_id="mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
    ...     max_tokens=10000,
    ... )
    >>> messages = [
    ...     {
    ...         "role": "user",
    ...         "content": [
    ...             {"type": "text", "text": "Explain quantum mechanics in simple terms."}
    ...         ]
    ...     }
    ... ]
    >>> response = engine(messages, stop_sequences=["END"])
    >>> print(response)
    "Quantum mechanics is the branch of physics that studies..."
    ```
    """

    def __init__(
        self,
        model_id: str,
        tool_name_key: str = "name",
        tool_arguments_key: str = "arguments",
        trust_remote_code: bool = False,
        **kwargs,
    ):
        super().__init__(
            flatten_messages_as_text=True, model_id=model_id, **kwargs
        )  # mlx-lm doesn't support vision models
        if not _is_package_available("mlx_lm"):
            raise ModuleNotFoundError(
                "Please install 'mlx-lm' extra to use 'MLXModel': `pip install 'smolagents[mlx-lm]'`"
            )
        import mlx_lm  # type: ignore

        self.model_id = model_id
        self.model, self.tokenizer = mlx_lm.load(model_id, tokenizer_config={"trust_remote_code": trust_remote_code})
        self.stream_generate = mlx_lm.stream_generate
        self.tool_name_key = tool_name_key
        self.tool_arguments_key = tool_arguments_key
        self.is_vlm = False  # mlx-lm doesn't support vision models

    def generate(
        self,
        messages: list[dict[str, str | list[dict]] | ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        if response_format is not None:
            raise ValueError("MLX does not support structured outputs.")
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )
        messages = completion_kwargs.pop("messages")
        stops = completion_kwargs.pop("stop", [])
        tools = completion_kwargs.pop("tools", None)
        completion_kwargs.pop("tool_choice", None)

        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
        )

        output_tokens = 0
        text = ""
        for response in self.stream_generate(self.model, self.tokenizer, prompt=prompt_ids, **completion_kwargs):
            output_tokens += 1
            text += response.text
            if any((stop_index := text.rfind(stop)) != -1 for stop in stops):
                text = text[:stop_index]
                break

        self._last_input_token_count = len(prompt_ids)
        self._last_output_token_count = output_tokens
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=text,
            raw={"out": text, "completion_kwargs": completion_kwargs},
            token_usage=TokenUsage(
                input_tokens=len(prompt_ids),
                output_tokens=output_tokens,
            ),
        )


class TransformersModel(Model):
    """A class that uses Hugging Face's Transformers library for language model interaction.

    This model allows you to load and use Hugging Face's models locally using the Transformers library. It supports features like stop sequences and grammar customization.

    > [!TIP]
    > You must have `transformers` and `torch` installed on your machine. Please run `pip install smolagents[transformers]` if it's not the case.

    Parameters:
        model_id (`str`):
            The Hugging Face model ID to be used for inference. This can be a path or model identifier from the Hugging Face model hub.
            For example, `"Qwen/Qwen2.5-Coder-32B-Instruct"`.
        device_map (`str`, *optional*):
            The device_map to initialize your model with.
        torch_dtype (`str`, *optional*):
            The torch_dtype to initialize your model with.
        trust_remote_code (bool, default `False`):
            Some models on the Hub require running remote code: for this model, you would have to set this flag to True.
        kwargs (dict, *optional*):
            Any additional keyword arguments that you want to use in model.generate(), for instance `max_new_tokens` or `device`.
        **kwargs:
            Additional keyword arguments to pass to `model.generate()`, for instance `max_new_tokens` or `device`.
    Raises:
        ValueError:
            If the model name is not provided.

    Example:
    ```python
    >>> engine = TransformersModel(
    ...     model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    ...     device="cuda",
    ...     max_new_tokens=5000,
    ... )
    >>> messages = [{"role": "user", "content": "Explain quantum mechanics in simple terms."}]
    >>> response = engine(messages, stop_sequences=["END"])
    >>> print(response)
    "Quantum mechanics is the branch of physics that studies..."
    ```
    """

    def __init__(
        self,
        model_id: str | None = None,
        device_map: str | None = None,
        torch_dtype: str | None = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoModelForImageTextToText,
                AutoProcessor,
                AutoTokenizer,
                TextIteratorStreamer,
            )
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install 'transformers' extra to use 'TransformersModel': `pip install 'smolagents[transformers]'`"
            )

        if not model_id:
            warnings.warn(
                "The 'model_id' parameter will be required in version 2.0.0. "
                "Please update your code to pass this parameter to avoid future errors. "
                "For now, it defaults to 'HuggingFaceTB/SmolLM2-1.7B-Instruct'.",
                FutureWarning,
            )
            model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

        default_max_tokens = 4096
        max_new_tokens = kwargs.get("max_new_tokens") or kwargs.get("max_tokens")
        if not max_new_tokens:
            kwargs["max_new_tokens"] = default_max_tokens
            logger.warning(
                f"`max_new_tokens` not provided, using this default value for `max_new_tokens`: {default_max_tokens}"
            )

        if device_map is None:
            device_map = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device_map}")
        self._is_vlm = False
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
            self._is_vlm = True
            self.streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)  # type: ignore

        except ValueError as e:
            if "Unrecognized configuration class" in str(e):
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    trust_remote_code=trust_remote_code,
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
                self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)  # type: ignore
            else:
                raise e
        except Exception as e:
            raise ValueError(f"Failed to load tokenizer and model for {model_id=}: {e}") from e
        super().__init__(flatten_messages_as_text=not self._is_vlm, model_id=model_id, **kwargs)

    def make_stopping_criteria(self, stop_sequences: list[str], tokenizer) -> "StoppingCriteriaList":
        """
        创建停止条件列表
        
        参数:
            stop_sequences (list[str]): 停止序列列表
            tokenizer: 分词器对象
            
        返回:
            StoppingCriteriaList: 停止条件列表
        """
        from transformers import StoppingCriteria, StoppingCriteriaList

        class StopOnStrings(StoppingCriteria):
            def __init__(self, stop_strings: list[str], tokenizer):
                """
                初始化字符串停止条件
                
                参数:
                    stop_strings (list[str]): 停止字符串列表
                    tokenizer: 分词器对象
                """
                self.stop_strings = stop_strings
                self.tokenizer = tokenizer
                self.stream = ""

            def reset(self):
                """重置流状态"""
                self.stream = ""

            def __call__(self, input_ids, scores, **kwargs):
                """
                检查是否应该停止生成
                
                参数:
                    input_ids: 输入token ID
                    scores: 分数
                    **kwargs: 其他参数
                    
                返回:
                    bool: 如果应该停止则返回True
                """
                generated = self.tokenizer.decode(input_ids[0][-1], skip_special_tokens=True)
                self.stream += generated
                if any([self.stream.endswith(stop_string) for stop_string in self.stop_strings]):
                    return True
                return False

        return StoppingCriteriaList([StopOnStrings(stop_sequences, tokenizer)])

    def _prepare_completion_args(
        self,
        messages: list[dict[str, str | list[dict]]],
        stop_sequences: list[str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        为Transformers模型准备生成参数
        
        参数:
            messages (list[dict]): 消息列表
            stop_sequences (list[str], 可选): 停止序列
            tools_to_call_from (list[Tool], 可选): 可调用的工具列表
            **kwargs: 其他关键字参数
            
        返回:
            dict: 准备好的生成参数
        """
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            **kwargs,
        )

        messages = completion_kwargs.pop("messages")
        stop_sequences = completion_kwargs.pop("stop", None)

        max_new_tokens = (
            kwargs.get("max_new_tokens")
            or kwargs.get("max_tokens")
            or self.kwargs.get("max_new_tokens")
            or self.kwargs.get("max_tokens")
            or 1024
        )
        prompt_tensor = (self.processor if hasattr(self, "processor") else self.tokenizer).apply_chat_template(
            messages,  # type: ignore
            tools=[get_tool_json_schema(tool) for tool in tools_to_call_from] if tools_to_call_from else None,
            return_tensors="pt",
            add_generation_prompt=True if tools_to_call_from else False,
            tokenize=True,
            return_dict=True,
        )
        prompt_tensor = prompt_tensor.to(self.model.device)  # type: ignore
        if hasattr(prompt_tensor, "input_ids"):
            prompt_tensor = prompt_tensor["input_ids"]

        model_tokenizer = self.processor.tokenizer if hasattr(self, "processor") else self.tokenizer
        stopping_criteria = (
            self.make_stopping_criteria(stop_sequences, tokenizer=model_tokenizer) if stop_sequences else None
        )
        completion_kwargs["max_new_tokens"] = max_new_tokens
        return dict(
            inputs=prompt_tensor,
            use_cache=True,
            stopping_criteria=stopping_criteria,
            **completion_kwargs,
        )

    def generate(
        self,
        messages: list[dict[str, str | list[dict]] | ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        if response_format is not None:
            raise ValueError("Transformers does not support structured outputs, use VLLMModel for this.")
        generation_kwargs = self._prepare_completion_args(
            messages=messages,
            stop_sequences=stop_sequences,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )
        count_prompt_tokens = generation_kwargs["inputs"].shape[1]  # type: ignore
        out = self.model.generate(
            **generation_kwargs,
        )
        generated_tokens = out[0, count_prompt_tokens:]
        if hasattr(self, "processor"):
            output_text = self.processor.decode(generated_tokens, skip_special_tokens=True)
        else:
            output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        if stop_sequences is not None:
            output_text = remove_stop_sequences(output_text, stop_sequences)

        self._last_input_token_count = count_prompt_tokens
        self._last_output_token_count = len(generated_tokens)
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=output_text,
            raw={
                "out": output_text,
                "completion_kwargs": {key: value for key, value in generation_kwargs.items() if key != "inputs"},
            },
            token_usage=TokenUsage(
                input_tokens=count_prompt_tokens,
                output_tokens=len(generated_tokens),
            ),
        )

    def generate_stream(
        self,
        messages: list[dict[str, str | list[dict]]],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> Generator[ChatMessageStreamDelta]:
        if response_format is not None:
            raise ValueError("Transformers does not support structured outputs, use VLLMModel for this.")
        generation_kwargs = self._prepare_completion_args(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )
        count_prompt_tokens = generation_kwargs["inputs"].shape[1]  # type: ignore

        thread = Thread(target=self.model.generate, kwargs={"streamer": self.streamer, **generation_kwargs})
        thread.start()

        # Generate with streaming
        for new_text in self.streamer:
            self._last_input_token_count = count_prompt_tokens
            self._last_output_token_count = 1
            yield ChatMessageStreamDelta(
                content=new_text,
                tool_calls=None,
                token_usage=TokenUsage(input_tokens=count_prompt_tokens, output_tokens=1),
            )
        thread.join()


class ApiModel(Model):
    """
    API 模型基类
    
    为基于外部 API 的语言模型提供通用基础框架。
    该类处理与远程 API 服务交互的通用功能，包括客户端管理、
    角色映射和连接配置等。
    
    核心功能:
    - 统一的 API 客户端管理
    - 自定义角色转换支持
    - 连接配置和认证处理
    - 错误处理和重试机制
    - 标准化的请求/响应处理
    
    设计模式:
    - 模板方法模式：定义通用流程，子类实现具体细节
    - 工厂模式：create_client 方法由子类实现
    - 策略模式：支持不同的角色转换策略
    
    子类实现要求:
    - 必须实现 create_client() 方法
    - 可以重写角色转换逻辑
    - 可以自定义请求参数处理
    
    支持的 API 类型:
    - REST API（如 OpenAI、HuggingFace）
    - 自定义 HTTP 服务
    - 云服务提供商 API
    
    参数:
        model_id (str): API 中使用的模型标识符
        custom_role_conversions (dict[str, str], 可选): 
            内部角色名称与 API 特定角色名称之间的转换映射
        client (Any, 可选): 预配置的 API 客户端实例，
            如果未提供将创建默认客户端
        **kwargs: 传递给父类的其他关键字参数
    """

    def __init__(
        self, model_id: str, custom_role_conversions: dict[str, str] | None = None, client: Any | None = None, **kwargs
    ):
        super().__init__(model_id=model_id, **kwargs)
        self.custom_role_conversions = custom_role_conversions or {}
        self.client = client or self.create_client()

    def create_client(self):
        """
        为特定服务创建 API 客户端 - 抽象方法
        
        这是一个抽象工厂方法，子类必须实现此方法来创建对应服务的 API 客户端。
        不同的子类将创建不同类型的客户端，如 OpenAI 客户端、HuggingFace 客户端等。
        
        实现要求:
        - 使用 self.client_kwargs 中的配置参数
        - 处理认证和连接配置
        - 返回可用的客户端实例
        - 确保客户端线程安全
        
        常见客户端类型:
        - openai.OpenAI: OpenAI API 客户端
        - InferenceClient: HuggingFace 推理客户端
        - boto3.client: AWS Bedrock 客户端
        - litellm: LiteLLM 统一客户端
        
        异常:
            NotImplementedError: 子类必须实现此方法
            
        返回:
            Any: 配置好的 API 客户端实例，具体类型由子类决定
        """
        raise NotImplementedError("Subclasses must implement this method to create a client")


class LiteLLMModel(ApiModel):
    """Model to use [LiteLLM Python SDK](https://docs.litellm.ai/docs/#litellm-python-sdk) to access hundreds of LLMs.

    Parameters:
        model_id (`str`):
            The model identifier to use on the server (e.g. "gpt-3.5-turbo").
        api_base (`str`, *optional*):
            The base URL of the provider API to call the model.
        api_key (`str`, *optional*):
            The API key to use for authentication.
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        flatten_messages_as_text (`bool`, *optional*): Whether to flatten messages as text.
            Defaults to `True` for models that start with "ollama", "groq", "cerebras".
        **kwargs:
            Additional keyword arguments to pass to the OpenAI API.
    """

    def __init__(
        self,
        model_id: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        flatten_messages_as_text: bool | None = None,
        **kwargs,
    ):
        if not model_id:
            warnings.warn(
                "The 'model_id' parameter will be required in version 2.0.0. "
                "Please update your code to pass this parameter to avoid future errors. "
                "For now, it defaults to 'anthropic/claude-3-5-sonnet-20240620'.",
                FutureWarning,
            )
            model_id = "anthropic/claude-3-5-sonnet-20240620"
        self.api_base = api_base
        self.api_key = api_key
        flatten_messages_as_text = (
            flatten_messages_as_text
            if flatten_messages_as_text is not None
            else model_id.startswith(("ollama", "groq", "cerebras"))
        )
        super().__init__(
            model_id=model_id,
            custom_role_conversions=custom_role_conversions,
            flatten_messages_as_text=flatten_messages_as_text,
            **kwargs,
        )

    def create_client(self):
        """
        创建LiteLLM客户端
        
        返回:
            litellm: LiteLLM模块对象
            
        异常:
            ModuleNotFoundError: 当litellm模块未安装时抛出
        """
        try:
            import litellm
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'litellm' extra to use LiteLLMModel: `pip install 'smolagents[litellm]'`"
            ) from e

        return litellm

    def generate(
        self,
        messages: list[dict[str, str | list[dict]] | ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            api_base=self.api_base,
            api_key=self.api_key,
            convert_images_to_image_urls=True,
            custom_role_conversions=self.custom_role_conversions,
            **kwargs,
        )

        response = self.client.completion(**completion_kwargs)

        self._last_input_token_count = response.usage.prompt_tokens
        self._last_output_token_count = response.usage.completion_tokens
        return ChatMessage.from_dict(
            response.choices[0].message.model_dump(include={"role", "content", "tool_calls"}),
            raw=response,
            token_usage=TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )

    def generate_stream(
        self,
        messages: list[dict[str, str | list[dict]]],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> Generator[ChatMessageStreamDelta]:
        if tools_to_call_from:
            raise NotImplementedError("Streaming is not yet supported for tool calling")
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            api_base=self.api_base,
            api_key=self.api_key,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )
        for event in self.client.completion(**completion_kwargs, stream=True, stream_options={"include_usage": True}):
            if event.choices:
                if event.choices[0].delta.content:
                    yield ChatMessageStreamDelta(
                        content=event.choices[0].delta.content,
                    )
            if getattr(event, "usage", None):
                self._last_input_token_count = event.usage.prompt_tokens
                self._last_output_token_count = event.usage.completion_tokens
                yield ChatMessageStreamDelta(
                    content="",
                    token_usage=TokenUsage(
                        input_tokens=event.usage.prompt_tokens,
                        output_tokens=event.usage.completion_tokens,
                    ),
                )


class LiteLLMRouterModel(LiteLLMModel):
    """Router‑based client for interacting with the [LiteLLM Python SDK Router](https://docs.litellm.ai/docs/routing).

    This class provides a high-level interface for distributing requests among multiple language models using
    the LiteLLM SDK's routing capabilities. It is responsible for initializing and configuring the router client,
    applying custom role conversions, and managing message formatting to ensure seamless integration with various LLMs.

    Parameters:
        model_id (`str`):
            Identifier for the model group to use from the model list (e.g., "model-group-1").
        model_list (`list[dict[str, Any]]`):
            Model configurations to be used for routing.
            Each configuration should include the model group name and any necessary parameters.
            For more details, refer to the [LiteLLM Routing](https://docs.litellm.ai/docs/routing#quick-start) documentation.
        client_kwargs (`dict[str, Any]`, *optional*):
            Additional configuration parameters for the Router client. For more details, see the
            [LiteLLM Routing Configurations](https://docs.litellm.ai/docs/routing).
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        flatten_messages_as_text (`bool`, *optional*): Whether to flatten messages as text.
            Defaults to `True` for models that start with "ollama", "groq", "cerebras".
        **kwargs:
            Additional keyword arguments to pass to the LiteLLM Router completion method.

    Example:
    ```python
    >>> import os
    >>> from smolagents import CodeAgent, WebSearchTool, LiteLLMRouterModel
    >>> os.environ["OPENAI_API_KEY"] = ""
    >>> os.environ["AWS_ACCESS_KEY_ID"] = ""
    >>> os.environ["AWS_SECRET_ACCESS_KEY"] = ""
    >>> os.environ["AWS_REGION"] = ""
    >>> llm_loadbalancer_model_list = [
    ...     {
    ...         "model_name": "model-group-1",
    ...         "litellm_params": {
    ...             "model": "gpt-4o-mini",
    ...             "api_key": os.getenv("OPENAI_API_KEY"),
    ...         },
    ...     },
    ...     {
    ...         "model_name": "model-group-1",
    ...         "litellm_params": {
    ...             "model": "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    ...             "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
    ...             "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
    ...             "aws_region_name": os.getenv("AWS_REGION"),
    ...         },
    ...     },
    >>> ]
    >>> model = LiteLLMRouterModel(
    ...    model_id="model-group-1",
    ...    model_list=llm_loadbalancer_model_list,
    ...    client_kwargs={
    ...        "routing_strategy":"simple-shuffle"
    ...    }
    >>> )
    >>> agent = CodeAgent(tools=[WebSearchTool()], model=model)
    >>> agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
    ```
    """

    def __init__(
        self,
        model_id: str,
        model_list: list[dict[str, Any]],
        client_kwargs: dict[str, Any] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        flatten_messages_as_text: bool | None = None,
        **kwargs,
    ):
        self.client_kwargs = {
            "model_list": model_list,
            **(client_kwargs or {}),
        }
        super().__init__(
            model_id=model_id,
            custom_role_conversions=custom_role_conversions,
            flatten_messages_as_text=flatten_messages_as_text,
            **kwargs,
        )

    def create_client(self):
        """
        创建LiteLLM路由器客户端
        
        返回:
            Router: 配置好的LiteLLM路由器实例
            
        异常:
            ModuleNotFoundError: 当litellm模块未安装时抛出
        """
        try:
            from litellm import Router
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'litellm' extra to use LiteLLMRouterModel: `pip install 'smolagents[litellm]'`"
            ) from e
        return Router(**self.client_kwargs)


class InferenceClientModel(ApiModel):
    """A class to interact with Hugging Face's Inference Providers for language model interaction.

    This model allows you to communicate with Hugging Face's models using Inference Providers. It can be used in both serverless mode, with a dedicated endpoint, or even with a local URL, supporting features like stop sequences and grammar customization.

    Providers include Cerebras, Cohere, Fal, Fireworks, HF-Inference, Hyperbolic, Nebius, Novita, Replicate, SambaNova, Together, and more.

    Parameters:
        model_id (`str`, *optional*, default `"Qwen/Qwen2.5-Coder-32B-Instruct"`):
            The Hugging Face model ID to be used for inference.
            This can be a model identifier from the Hugging Face model hub or a URL to a deployed Inference Endpoint.
            Currently, it defaults to `"Qwen/Qwen2.5-Coder-32B-Instruct"`, but this may change in the future.
        provider (`str`, *optional*):
            Name of the provider to use for inference. A list of supported providers can be found in the [Inference Providers documentation](https://huggingface.co/docs/inference-providers/index#partners).
            Defaults to "auto" i.e. the first of the providers available for the model, sorted by the user's order [here](https://hf.co/settings/inference-providers).
            If `base_url` is passed, then `provider` is not used.
        token (`str`, *optional*):
            Token used by the Hugging Face API for authentication. This token need to be authorized 'Make calls to the serverless Inference Providers'.
            If the model is gated (like Llama-3 models), the token also needs 'Read access to contents of all public gated repos you can access'.
            If not provided, the class will try to use environment variable 'HF_TOKEN', else use the token stored in the Hugging Face CLI configuration.
        timeout (`int`, *optional*, defaults to 120):
            Timeout for the API request, in seconds.
        client_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the Hugging Face InferenceClient.
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        api_key (`str`, *optional*):
            Token to use for authentication. This is a duplicated argument from `token` to make [`InferenceClientModel`]
            follow the same pattern as `openai.OpenAI` client. Cannot be used if `token` is set. Defaults to None.
        bill_to (`str`, *optional*):
            The billing account to use for the requests. By default the requests are billed on the user's account. Requests can only be billed to
            an organization the user is a member of, and which has subscribed to Enterprise Hub.
        base_url (`str`, `optional`):
            Base URL to run inference. This is a duplicated argument from `model` to make [`InferenceClientModel`]
            follow the same pattern as `openai.OpenAI` client. Cannot be used if `model` is set. Defaults to None.
        **kwargs:
            Additional keyword arguments to pass to the Hugging Face InferenceClient.

    Raises:
        ValueError:
            If the model name is not provided.

    Example:
    ```python
    >>> engine = InferenceClientModel(
    ...     model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    ...     provider="nebius",
    ...     token="your_hf_token_here",
    ...     max_tokens=5000,
    ... )
    >>> messages = [{"role": "user", "content": "Explain quantum mechanics in simple terms."}]
    >>> response = engine(messages, stop_sequences=["END"])
    >>> print(response)
    "Quantum mechanics is the branch of physics that studies..."
    ```
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
        provider: str | None = None,
        token: str | None = None,
        timeout: int = 120,
        client_kwargs: dict[str, Any] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        api_key: str | None = None,
        bill_to: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        if token is not None and api_key is not None:
            raise ValueError(
                "Received both `token` and `api_key` arguments. Please provide only one of them."
                " `api_key` is an alias for `token` to make the API compatible with OpenAI's client."
                " It has the exact same behavior as `token`."
            )
        token = token if token is not None else api_key
        if token is None:
            token = os.getenv("HF_TOKEN")
        self.client_kwargs = {
            **(client_kwargs or {}),
            "model": model_id,
            "provider": provider,
            "token": token,
            "timeout": timeout,
            "bill_to": bill_to,
            "base_url": base_url,
        }
        super().__init__(model_id=model_id, custom_role_conversions=custom_role_conversions, **kwargs)

    def create_client(self):
        """
        创建Hugging Face推理客户端
        
        返回:
            InferenceClient: 配置好的Hugging Face推理客户端实例
        """
        from huggingface_hub import InferenceClient

        return InferenceClient(**self.client_kwargs)

    def generate(
        self,
        messages: list[dict[str, str | list[dict]] | ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        if response_format is not None and self.client_kwargs["provider"] not in STRUCTURED_GENERATION_PROVIDERS:
            raise ValueError(
                "InferenceClientModel only supports structured outputs with these providers:"
                + ", ".join(STRUCTURED_GENERATION_PROVIDERS)
            )
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            tools_to_call_from=tools_to_call_from,
            # response_format=response_format,
            convert_images_to_image_urls=True,
            custom_role_conversions=self.custom_role_conversions,
            **kwargs,
        )
        response = self.client.chat_completion(**completion_kwargs)

        self._last_input_token_count = response.usage.prompt_tokens
        self._last_output_token_count = response.usage.completion_tokens
        return ChatMessage.from_dict(
            asdict(response.choices[0].message),
            raw=response,
            token_usage=TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )

    def generate_stream(
        self,
        messages: list[dict[str, str | list[dict]]],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> Generator[ChatMessageStreamDelta]:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )

        # Track accumulated tool calls and content
        accumulated_tool_calls = {}
        current_content = ""

        for event in self.client.chat.completions.create(
            **completion_kwargs, stream=True, stream_options={"include_usage": True}
        ):
            if event.choices:
                choice = event.choices[0]
                if choice.delta is None:
                    if not getattr(choice, "finish_reason", None):
                        raise ValueError(f"No content or tool calls in event: {event}")
                else:
                    delta = choice.delta

                    # Handle content streaming
                    if delta.content:
                        current_content += delta.content
                        yield ChatMessageStreamDelta(content=delta.content)

                    # Handle tool call streaming
                    if delta.tool_calls:
                        for tool_call_delta in delta.tool_calls:  # ?ormally there should be only one call at a time
                            # Extend accumulated_tool_calls list to accommodate the new tool call if needed
                            while len(accumulated_tool_calls) <= tool_call_delta.index:
                                accumulated_tool_calls[tool_call_delta.index] = {
                                    "id": None,
                                    "type": None,
                                    "function": {"name": None, "arguments": ""},
                                }

                            # Update the tool call at the specific index
                            tool_call = accumulated_tool_calls[tool_call_delta.index]

                            if tool_call_delta.id:
                                tool_call["id"] = tool_call_delta.index
                            if tool_call_delta.type:
                                tool_call["type"] = tool_call_delta.type
                            if tool_call_delta.function:
                                if tool_call_delta.function.name:
                                    tool_call["function"]["name"] = tool_call_delta.function.name
                                if tool_call_delta.function.arguments:
                                    tool_call["function"]["arguments"] += tool_call_delta.function.arguments

                        yield ChatMessageStreamDelta(
                            content=current_content,
                            tool_calls=[
                                ToolCallStreamDelta(
                                    id=tool_call["id"],
                                    type=tool_call["type"],
                                    function=ChatMessageToolCallDefinition(
                                        name=tool_call["function"]["name"],
                                        arguments=tool_call["function"]["arguments"],
                                    ),
                                )
                                for tool_call in accumulated_tool_calls.values()
                            ],
                        )

            if event.usage:
                self._last_input_token_count = event.usage.prompt_tokens
                self._last_output_token_count = event.usage.completion_tokens
                yield ChatMessageStreamDelta(
                    content="",
                    token_usage=TokenUsage(
                        input_tokens=event.usage.prompt_tokens,
                        output_tokens=event.usage.completion_tokens,
                    ),
                )


class HfApiModel(InferenceClientModel):
    def __new__(cls, *args, **kwargs):
        """
        创建HfApiModel实例（已弃用）
        
        HfApiModel已在版本1.14.0中重命名为InferenceClientModel，
        将在版本1.17.0中移除。
        
        参数:
            *args: 位置参数
            **kwargs: 关键字参数
            
        返回:
            InferenceClientModel实例
        """
        warnings.warn(
            "HfApiModel was renamed to InferenceClientModel in version 1.14.0 and will be removed in 1.17.0.",
            FutureWarning,
        )
        return super().__new__(cls)


class OpenAIServerModel(ApiModel):
    """
    OpenAI 兼容服务器模型
    
    连接到 OpenAI 兼容的 API 服务器，支持原生 OpenAI API 和各种兼容服务。
    这包括官方 OpenAI API、自建的兼容服务器、以及第三方 OpenAI 兼容服务。
    
    支持的服务类型:
    - OpenAI 官方 API (api.openai.com)
    - Azure OpenAI 服务
    - 本地部署的兼容服务器
    - 第三方 OpenAI 兼容 API
    - 自建的模型服务
    
    主要特性:
    - 完整的 OpenAI API 兼容性
    - 支持聊天完成和工具调用
    - 流式响应和批量处理
    - 自定义角色转换
    - 组织和项目管理
    - 错误处理和重试机制
    
    使用场景:
    - 连接 OpenAI 官方服务
    - 使用企业内部的模型服务
    - 测试和开发环境
    - 多模型服务切换
    
    参数:
        model_id (str): 服务器上使用的模型标识符，如 "gpt-3.5-turbo"
        api_base (str, 可选): OpenAI 兼容 API 服务器的基础 URL
        api_key (str, 可选): 用于身份验证的 API 密钥
        organization (str, 可选): API 请求使用的组织标识
        project (str, 可选): API 请求使用的项目标识
        client_kwargs (dict[str, Any], 可选): 传递给 OpenAI 客户端的额外参数，
            如 organization、project、max_retries 等
        custom_role_conversions (dict[str, str], 可选): 自定义角色转换映射，
            用于转换消息角色，适用于不支持特定角色（如 "system"）的模型
        flatten_messages_as_text (bool, 默认 False): 是否将消息扁平化为文本
        **kwargs: 传递给 OpenAI API 的其他关键字参数
    """

    def __init__(
        self,
        model_id: str,
        api_base: str | None = None,
        api_key: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        client_kwargs: dict[str, Any] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        flatten_messages_as_text: bool = False,
        **kwargs,
    ):
        self.client_kwargs = {
            **(client_kwargs or {}),
            "api_key": api_key,
            "base_url": api_base,
            "organization": organization,
            "project": project,
        }
        super().__init__(
            model_id=model_id,
            custom_role_conversions=custom_role_conversions,
            flatten_messages_as_text=flatten_messages_as_text,
            **kwargs,
        )

    def create_client(self):
        """
        创建OpenAI客户端
        
        返回:
            openai.OpenAI: 配置好的OpenAI客户端实例
            
        异常:
            ModuleNotFoundError: 当openai模块未安装时抛出
        """
        try:
            import openai
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'openai' extra to use OpenAIServerModel: `pip install 'smolagents[openai]'`"
            ) from e

        return openai.OpenAI(**self.client_kwargs)

    def generate_stream(
        self,
        messages: list[dict[str, str | list[dict]]],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> Generator[ChatMessageStreamDelta]:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )

        # Track accumulated tool calls and content
        accumulated_tool_calls = {}
        current_content = ""

        for event in self.client.chat.completions.create(
            **completion_kwargs, stream=True, stream_options={"include_usage": True}
        ):
            if event.choices:
                choice = event.choices[0]
                if choice.delta is None:
                    if not getattr(choice, "finish_reason", None):
                        raise ValueError(f"No content or tool calls in event: {event}")
                else:
                    delta = choice.delta

                    # Handle content streaming
                    if delta.content:
                        current_content += delta.content
                        yield ChatMessageStreamDelta(content=delta.content)

                    # Handle tool call streaming
                    if delta.tool_calls:
                        for tool_call_delta in delta.tool_calls:  # ?ormally there should be only one call at a time
                            # Extend accumulated_tool_calls list to accommodate the new tool call if needed
                            while len(accumulated_tool_calls) <= tool_call_delta.index:
                                accumulated_tool_calls[tool_call_delta.index] = {
                                    "id": None,
                                    "type": None,
                                    "function": {"name": None, "arguments": ""},
                                }

                            # Update the tool call at the specific index
                            tool_call = accumulated_tool_calls[tool_call_delta.index]

                            if tool_call_delta.id:
                                tool_call["id"] = tool_call_delta.index
                            if tool_call_delta.type:
                                tool_call["type"] = tool_call_delta.type
                            if tool_call_delta.function:
                                if tool_call_delta.function.name:
                                    tool_call["function"]["name"] = tool_call_delta.function.name
                                if tool_call_delta.function.arguments:
                                    tool_call["function"]["arguments"] += tool_call_delta.function.arguments

                        yield ChatMessageStreamDelta(
                            content=current_content,
                            tool_calls=[
                                ToolCallStreamDelta(
                                    id=tool_call["id"],
                                    type=tool_call["type"],
                                    function=ChatMessageToolCallDefinition(
                                        name=tool_call["function"]["name"],
                                        arguments=tool_call["function"]["arguments"],
                                    ),
                                )
                                for tool_call in accumulated_tool_calls.values()
                            ],
                        )

            if event.usage:
                self._last_input_token_count = event.usage.prompt_tokens
                self._last_output_token_count = event.usage.completion_tokens
                yield ChatMessageStreamDelta(
                    content="",
                    token_usage=TokenUsage(
                        input_tokens=event.usage.prompt_tokens,
                        output_tokens=event.usage.completion_tokens,
                    ),
                )

    def generate(
        self,
        messages: list[dict[str, str | list[dict]] | ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )
        response = self.client.chat.completions.create(**completion_kwargs)

        self._last_input_token_count = response.usage.prompt_tokens
        self._last_output_token_count = response.usage.completion_tokens
        return ChatMessage.from_dict(
            response.choices[0].message.model_dump(include={"role", "content", "tool_calls"}),
            raw=response,
            token_usage=TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )


class AzureOpenAIServerModel(OpenAIServerModel):
    """This model connects to an Azure OpenAI deployment.

    Parameters:
        model_id (`str`):
            The model deployment name to use when connecting (e.g. "gpt-4o-mini").
        azure_endpoint (`str`, *optional*):
            The Azure endpoint, including the resource, e.g. `https://example-resource.azure.openai.com/`. If not provided, it will be inferred from the `AZURE_OPENAI_ENDPOINT` environment variable.
        api_key (`str`, *optional*):
            The API key to use for authentication. If not provided, it will be inferred from the `AZURE_OPENAI_API_KEY` environment variable.
        api_version (`str`, *optional*):
            The API version to use. If not provided, it will be inferred from the `OPENAI_API_VERSION` environment variable.
        client_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the AzureOpenAI client (like organization, project, max_retries etc.).
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        **kwargs:
            Additional keyword arguments to pass to the Azure OpenAI API.
    """

    def __init__(
        self,
        model_id: str,
        azure_endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        client_kwargs: dict[str, Any] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        **kwargs,
    ):
        client_kwargs = client_kwargs or {}
        client_kwargs.update(
            {
                "api_version": api_version,
                "azure_endpoint": azure_endpoint,
            }
        )
        super().__init__(
            model_id=model_id,
            api_key=api_key,
            client_kwargs=client_kwargs,
            custom_role_conversions=custom_role_conversions,
            **kwargs,
        )

    def create_client(self):
        """
        创建Azure OpenAI客户端
        
        返回:
            openai.AzureOpenAI: 配置好的Azure OpenAI客户端实例
            
        异常:
            ModuleNotFoundError: 当openai模块未安装时抛出
        """
        try:
            import openai
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'openai' extra to use AzureOpenAIServerModel: `pip install 'smolagents[openai]'`"
            ) from e

        return openai.AzureOpenAI(**self.client_kwargs)


class AmazonBedrockServerModel(ApiModel):
    """
    A model class for interacting with Amazon Bedrock Server models through the Bedrock API.

    This class provides an interface to interact with various Bedrock language models,
    allowing for customized model inference, guardrail configuration, message handling,
    and other parameters allowed by boto3 API.

    Parameters:
        model_id (`str`):
            The model identifier to use on Bedrock (e.g. "us.amazon.nova-pro-v1:0").
        client (`boto3.client`, *optional*):
            A custom boto3 client for AWS interactions. If not provided, a default client will be created.
        client_kwargs (dict[str, Any], *optional*):
            Keyword arguments used to configure the boto3 client if it needs to be created internally.
            Examples include `region_name`, `config`, or `endpoint_url`.
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
            Defaults to converting all roles to "user" role to enable using all the Bedrock models.
        flatten_messages_as_text (`bool`, default `False`):
            Whether to flatten messages as text.
        **kwargs
            Additional keyword arguments passed directly to the underlying API calls.

    Example:
        Creating a model instance with default settings:
        >>> bedrock_model = AmazonBedrockServerModel(
        ...     model_id='us.amazon.nova-pro-v1:0'
        ... )

        Creating a model instance with a custom boto3 client:
        >>> import boto3
        >>> client = boto3.client('bedrock-runtime', region_name='us-west-2')
        >>> bedrock_model = AmazonBedrockServerModel(
        ...     model_id='us.amazon.nova-pro-v1:0',
        ...     client=client
        ... )

        Creating a model instance with client_kwargs for internal client creation:
        >>> bedrock_model = AmazonBedrockServerModel(
        ...     model_id='us.amazon.nova-pro-v1:0',
        ...     client_kwargs={'region_name': 'us-west-2', 'endpoint_url': 'https://custom-endpoint.com'}
        ... )

        Creating a model instance with inference and guardrail configurations:
        >>> additional_api_config = {
        ...     "inferenceConfig": {
        ...         "maxTokens": 3000
        ...     },
        ...     "guardrailConfig": {
        ...         "guardrailIdentifier": "identify1",
        ...         "guardrailVersion": 'v1'
        ...     },
        ... }
        >>> bedrock_model = AmazonBedrockServerModel(
        ...     model_id='anthropic.claude-3-haiku-20240307-v1:0',
        ...     **additional_api_config
        ... )
    """

    def __init__(
        self,
        model_id: str,
        client=None,
        client_kwargs: dict[str, Any] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        **kwargs,
    ):
        self.client_kwargs = client_kwargs or {}

        # Bedrock only supports `assistant` and `user` roles.
        # Many Bedrock models do not allow conversations to start with the `assistant` role, so the default is set to `user/user`.
        # This parameter is retained for future model implementations and extended support.
        custom_role_conversions = custom_role_conversions or {
            MessageRole.SYSTEM: MessageRole.USER,
            MessageRole.ASSISTANT: MessageRole.USER,
            MessageRole.TOOL_CALL: MessageRole.USER,
            MessageRole.TOOL_RESPONSE: MessageRole.USER,
        }

        super().__init__(
            model_id=model_id,
            custom_role_conversions=custom_role_conversions,
            flatten_messages_as_text=False,  # Bedrock API doesn't support flatten messages, must be a list of messages
            client=client,
            **kwargs,
        )

    def _prepare_completion_kwargs(
        self,
        messages: list[dict[str, str | list[dict]]],
        stop_sequences: list[str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        custom_role_conversions: dict[str, str] | None = None,
        convert_images_to_image_urls: bool = False,
        **kwargs,
    ) -> dict:
        """
        重写基础方法以处理Bedrock特定配置
        
        此实现调整完成关键字参数以符合Bedrock的要求，
        确保与其独特设置和约束的兼容性。
        
        参数:
            messages (list[dict]): 消息列表
            stop_sequences (list[str], 可选): 停止序列
            tools_to_call_from (list[Tool], 可选): 可调用的工具列表
            custom_role_conversions (dict, 可选): 自定义角色转换
            convert_images_to_image_urls (bool): 是否转换图像为URL
            **kwargs: 其他关键字参数
            
        返回:
            dict: 适配Bedrock格式的参数字典
        """
        completion_kwargs = super()._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=None,  # Bedrock support stop_sequence using Inference Config
            tools_to_call_from=tools_to_call_from,
            custom_role_conversions=custom_role_conversions,
            convert_images_to_image_urls=convert_images_to_image_urls,
            **kwargs,
        )

        # Not all models in Bedrock support `toolConfig`. Also, smolagents already include the tool call in the prompt,
        # so adding `toolConfig` could cause conflicts. We remove it to avoid issues.
        completion_kwargs.pop("toolConfig", None)

        # The Bedrock API does not support the `type` key in requests.
        # This block of code modifies the object to meet Bedrock's requirements.
        for message in completion_kwargs.get("messages", []):
            for content in message.get("content", []):
                if "type" in content:
                    del content["type"]

        return {
            "modelId": self.model_id,
            **completion_kwargs,
        }

    def create_client(self):
        """
        创建Amazon Bedrock客户端
        
        返回:
            boto3.client: 配置好的Bedrock运行时客户端实例
            
        异常:
            ModuleNotFoundError: 当boto3模块未安装时抛出
        """
        try:
            import boto3  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Please install 'bedrock' extra to use AmazonBedrockServerModel: `pip install 'smolagents[bedrock]'`"
            ) from e

        return boto3.client("bedrock-runtime", **self.client_kwargs)

    def generate(
        self,
        messages: list[dict[str, str | list[dict]] | ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        if response_format is not None:
            raise ValueError("Amazon Bedrock does not support response_format")
        completion_kwargs: dict = self._prepare_completion_kwargs(
            messages=messages,
            tools_to_call_from=tools_to_call_from,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )

        # self.client is created in ApiModel class
        response = self.client.converse(**completion_kwargs)

        # Get first message
        response["output"]["message"]["content"] = response["output"]["message"]["content"][0]["text"]

        self._last_input_token_count = response["usage"]["inputTokens"]
        self._last_output_token_count = response["usage"]["outputTokens"]
        return ChatMessage.from_dict(
            response["output"]["message"],
            raw=response,
            token_usage=TokenUsage(
                input_tokens=response["usage"]["inputTokens"],
                output_tokens=response["usage"]["outputTokens"],
            ),
        )


__all__ = [
    "MessageRole",
    "tool_role_conversions",
    "get_clean_message_list",
    "Model",
    "MLXModel",
    "TransformersModel",
    "ApiModel",
    "InferenceClientModel",
    "HfApiModel",
    "LiteLLMModel",
    "LiteLLMRouterModel",
    "OpenAIServerModel",
    "VLLMModel",
    "AzureOpenAIServerModel",
    "AmazonBedrockServerModel",
    "ChatMessage",
]
