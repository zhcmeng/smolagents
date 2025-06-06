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
工具函数模块 - smolagents 的核心工具函数库

本模块提供了 smolagents 框架中使用的各种实用工具函数，包括：
- 代码解析和提取
- JSON 处理和序列化
- 错误异常类定义
- 动态源码获取
- 图像编码处理
- 安全验证函数

主要功能:
- 从 LLM 输出中提取和解析代码块
- 处理工具调用的 JSON 格式数据
- 提供代理相关的异常类
- 支持动态环境（如 Jupyter）中的源码获取
- 图像的 base64 编码转换

作者: HuggingFace 团队
版本: 1.0
"""

import ast
import base64
import importlib.metadata
import importlib.util
import inspect
import json
import keyword
import os
import re
import types
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from smolagents.memory import AgentLogger


__all__ = ["AgentError"]


@lru_cache
def _is_package_available(package_name: str) -> bool:
    try:
        importlib.metadata.version(package_name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


BASE_BUILTIN_MODULES = [
    "collections",
    "datetime",
    "itertools",
    "math",
    "queue",
    "random",
    "re",
    "stat",
    "statistics",
    "time",
    "unicodedata",
]


def escape_code_brackets(text: str) -> str:
    """Escapes square brackets in code segments while preserving Rich styling tags."""

    def replace_bracketed_content(match):
        content = match.group(1)
        cleaned = re.sub(
            r"bold|red|green|blue|yellow|magenta|cyan|white|black|italic|dim|\s|#[0-9a-fA-F]{6}", "", content
        )
        return f"\\[{content}\\]" if cleaned.strip() else f"[{content}]"

    return re.sub(r"\[([^\]]*)\]", replace_bracketed_content, text)


class AgentError(Exception):
    """
    代理异常基类 - 所有代理相关异常的基础类
    
    该类为所有代理相关的异常提供统一的基础功能，包括错误消息记录和序列化。
    
    参数:
        message (str): 错误消息
        logger (AgentLogger): 用于记录错误的日志记录器
    """

    def __init__(self, message, logger: "AgentLogger"):
        super().__init__(message)
        self.message = message
        logger.log_error(message)

    def dict(self) -> dict[str, str]:
        """
        返回异常的字典表示
        
        返回:
            dict[str, str]: 包含异常类型和消息的字典
        """
        return {"type": self.__class__.__name__, "message": str(self.message)}


class AgentParsingError(AgentError):
    """代理解析错误 - 当代理无法正确解析 LLM 输出时抛出"""
    pass


class AgentExecutionError(AgentError):
    """代理执行错误 - 当代理执行过程中发生错误时抛出"""
    pass


class AgentMaxStepsError(AgentError):
    """代理最大步数错误 - 当代理达到最大执行步数限制时抛出"""
    pass


class AgentToolCallError(AgentExecutionError):
    """工具调用错误 - 当向工具传递错误参数时抛出"""
    pass


class AgentToolExecutionError(AgentExecutionError):
    """工具执行错误 - 当工具执行过程中发生错误时抛出"""
    pass


class AgentGenerationError(AgentError):
    """代理生成错误 - 当 LLM 生成过程中发生错误时抛出"""
    pass


def make_json_serializable(obj: Any) -> Any:
    """Recursive function to make objects JSON serializable"""
    if obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        # Try to parse string as JSON if it looks like a JSON object/array
        if isinstance(obj, str):
            try:
                if (obj.startswith("{") and obj.endswith("}")) or (obj.startswith("[") and obj.endswith("]")):
                    parsed = json.loads(obj)
                    return make_json_serializable(parsed)
            except json.JSONDecodeError:
                pass
        return obj
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):
        # For custom objects, convert their __dict__ to a serializable format
        return {"_type": obj.__class__.__name__, **{k: make_json_serializable(v) for k, v in obj.__dict__.items()}}
    else:
        # For any other type, convert to string
        return str(obj)


def parse_json_blob(json_blob: str) -> tuple[dict[str, str], str]:
    "Extracts the JSON blob from the input and returns the JSON data and the rest of the input."
    try:
        first_accolade_index = json_blob.find("{")
        last_accolade_index = [a.start() for a in list(re.finditer("}", json_blob))][-1]
        json_data = json_blob[first_accolade_index : last_accolade_index + 1]
        json_data = json.loads(json_data, strict=False)
        return json_data, json_blob[:first_accolade_index]
    except IndexError:
        raise ValueError("The model output does not contain any JSON blob.")
    except json.JSONDecodeError as e:
        place = e.pos
        if json_blob[place - 1 : place + 2] == "},\n":
            raise ValueError(
                "JSON is invalid: you probably tried to provide multiple tool calls in one action. PROVIDE ONLY ONE TOOL CALL."
            )
        raise ValueError(
            f"The JSON blob you used is invalid due to the following error: {e}.\n"
            f"JSON blob was: {json_blob}, decoding failed on that specific part of the blob:\n"
            f"'{json_blob[place - 4 : place + 5]}'."
        )


def extract_code_from_text(text: str) -> str | None:
    """Extract code from the LLM's output."""
    pattern = r"```(?:py|python)?\s*\n(.*?)\n```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return "\n\n".join(match.strip() for match in matches)
    return None


def parse_code_blobs(text: str) -> str:
    """Extract code blocs from the LLM's output.

    If a valid code block is passed, it returns it directly.

    Args:
        text (`str`): LLM's output text to parse.

    Returns:
        `str`: Extracted code block.

    Raises:
        ValueError: If no valid code block is found in the text.
    """
    matches = extract_code_from_text(text)
    if matches:
        return matches
    # Maybe the LLM outputted a code blob directly
    try:
        ast.parse(text)
        return text
    except SyntaxError:
        pass

    if "final" in text and "answer" in text:
        raise ValueError(
            dedent(
                f"""
                Your code snippet is invalid, because the regex pattern ```(?:py|python)?\\s*\\n(.*?)\\n``` was not found in it.
                Here is your code snippet:
                {text}
                It seems like you're trying to return the final answer, you can do it as follows:
                Code:
                ```py
                final_answer("YOUR FINAL ANSWER HERE")
                ```<end_code>
                """
            ).strip()
        )
    raise ValueError(
        dedent(
            f"""
            Your code snippet is invalid, because the regex pattern ```(?:py|python)?\\s*\\n(.*?)\\n``` was not found in it.
            Here is your code snippet:
            {text}
            Make sure to include code with the correct pattern, for instance:
            Thoughts: Your thoughts
            Code:
            ```py
            # Your python code here
            ```<end_code>
            """
        ).strip()
    )


MAX_LENGTH_TRUNCATE_CONTENT = 20000


def truncate_content(content: str, max_length: int = MAX_LENGTH_TRUNCATE_CONTENT) -> str:
    if len(content) <= max_length:
        return content
    else:
        return (
            content[: max_length // 2]
            + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
            + content[-max_length // 2 :]
        )


class ImportFinder(ast.NodeVisitor):
    def __init__(self):
        self.packages = set()

    def visit_Import(self, node):
        for alias in node.names:
            # Get the base package name (before any dots)
            base_package = alias.name.split(".")[0]
            self.packages.add(base_package)

    def visit_ImportFrom(self, node):
        if node.module:  # for "from x import y" statements
            # Get the base package name (before any dots)
            base_package = node.module.split(".")[0]
            self.packages.add(base_package)


def get_method_source(method):
    """Get source code for a method, including bound methods."""
    if isinstance(method, types.MethodType):
        method = method.__func__
    return get_source(method)


def is_same_method(method1, method2):
    """Compare two methods by their source code."""
    try:
        source1 = get_method_source(method1)
        source2 = get_method_source(method2)

        # Remove method decorators if any
        source1 = "\n".join(line for line in source1.split("\n") if not line.strip().startswith("@"))
        source2 = "\n".join(line for line in source2.split("\n") if not line.strip().startswith("@"))

        return source1 == source2
    except (TypeError, OSError):
        return False


def is_same_item(item1, item2):
    """Compare two class items (methods or attributes) for equality."""
    if callable(item1) and callable(item2):
        return is_same_method(item1, item2)
    else:
        return item1 == item2


def instance_to_source(instance, base_cls=None):
    """Convert an instance to its class source code representation."""
    cls = instance.__class__
    class_name = cls.__name__

    # Start building class lines
    class_lines = []
    if base_cls:
        class_lines.append(f"class {class_name}({base_cls.__name__}):")
    else:
        class_lines.append(f"class {class_name}:")

    # Add docstring if it exists and differs from base
    if cls.__doc__ and (not base_cls or cls.__doc__ != base_cls.__doc__):
        class_lines.append(f'    """{cls.__doc__}"""')

    # Add class-level attributes
    class_attrs = {
        name: value
        for name, value in cls.__dict__.items()
        if not name.startswith("__")
        and not callable(value)
        and not (base_cls and hasattr(base_cls, name) and getattr(base_cls, name) == value)
    }

    for name, value in class_attrs.items():
        if isinstance(value, str):
            # multiline value
            if "\n" in value:
                escaped_value = value.replace('"""', r"\"\"\"")  # Escape triple quotes
                class_lines.append(f'    {name} = """{escaped_value}"""')
            else:
                class_lines.append(f"    {name} = {json.dumps(value)}")
        else:
            class_lines.append(f"    {name} = {repr(value)}")

    if class_attrs:
        class_lines.append("")

    # Add methods
    methods = {
        name: func.__wrapped__ if hasattr(func, "__wrapped__") else func
        for name, func in cls.__dict__.items()
        if callable(func)
        and (
            not base_cls
            or not hasattr(base_cls, name)
            or (
                isinstance(func, (staticmethod, classmethod))
                or (getattr(base_cls, name).__code__.co_code != func.__code__.co_code)
            )
        )
    }

    for name, method in methods.items():
        method_source = get_source(method)
        # Clean up the indentation
        method_lines = method_source.split("\n")
        first_line = method_lines[0]
        indent = len(first_line) - len(first_line.lstrip())
        method_lines = [line[indent:] for line in method_lines]
        method_source = "\n".join(["    " + line if line.strip() else line for line in method_lines])
        class_lines.append(method_source)
        class_lines.append("")

    # Find required imports using ImportFinder
    import_finder = ImportFinder()
    import_finder.visit(ast.parse("\n".join(class_lines)))
    required_imports = import_finder.packages

    # Build final code with imports
    final_lines = []

    # Add base class import if needed
    if base_cls:
        final_lines.append(f"from {base_cls.__module__} import {base_cls.__name__}")

    # Add discovered imports
    for package in required_imports:
        final_lines.append(f"import {package}")

    if final_lines:  # Add empty line after imports
        final_lines.append("")

    # Add the class code
    final_lines.extend(class_lines)

    return "\n".join(final_lines)


def get_source(obj) -> str:
    """Get the source code of a class or callable object (e.g.: function, method).
    First attempts to get the source code using `inspect.getsource`.
    In a dynamic environment (e.g.: Jupyter, IPython), if this fails,
    falls back to retrieving the source code from the current interactive shell session.

    Args:
        obj: A class or callable object (e.g.: function, method)

    Returns:
        str: The source code of the object, dedented and stripped

    Raises:
        TypeError: If object is not a class or callable
        OSError: If source code cannot be retrieved from any source
        ValueError: If source cannot be found in IPython history

    Note:
        TODO: handle Python standard REPL
    """
    if not (isinstance(obj, type) or callable(obj)):
        raise TypeError(f"Expected class or callable, got {type(obj)}")

    inspect_error = None
    try:
        # Handle dynamically created classes
        source = getattr(obj, "__source__", None) or inspect.getsource(obj)
        return dedent(source).strip()
    except OSError as e:
        # let's keep track of the exception to raise it if all further methods fail
        inspect_error = e
    try:
        import IPython

        shell = IPython.get_ipython()
        if not shell:
            raise ImportError("No active IPython shell found")
        all_cells = "\n".join(shell.user_ns.get("In", [])).strip()
        if not all_cells:
            raise ValueError("No code cells found in IPython session")

        tree = ast.parse(all_cells)
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == obj.__name__:
                return dedent("\n".join(all_cells.split("\n")[node.lineno - 1 : node.end_lineno])).strip()
        raise ValueError(f"Could not find source code for {obj.__name__} in IPython history")
    except ImportError:
        # IPython is not available, let's just raise the original inspect error
        raise inspect_error
    except ValueError as e:
        # IPython is available but we couldn't find the source code, let's raise the error
        raise e from inspect_error


def encode_image_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def make_image_url(base64_image):
    return f"data:image/png;base64,{base64_image}"


def make_init_file(folder: str | Path):
    os.makedirs(folder, exist_ok=True)
    # Create __init__
    with open(os.path.join(folder, "__init__.py"), "w"):
        pass


def is_valid_name(name: str) -> bool:
    return name.isidentifier() and not keyword.iskeyword(name) if isinstance(name, str) else False
