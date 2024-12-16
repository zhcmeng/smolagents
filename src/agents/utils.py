#!/usr/bin/env python
# coding=utf-8

# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
import json
import re
from typing import Tuple, Dict, Union
import ast
from rich.console import Console

from transformers.utils.import_utils import _is_package_available

_pygments_available = _is_package_available("pygments")


def is_pygments_available():
    return _pygments_available


console = Console()


def parse_json_blob(json_blob: str) -> Dict[str, str]:
    try:
        first_accolade_index = json_blob.find("{")
        last_accolade_index = [a.start() for a in list(re.finditer("}", json_blob))][-1]
        json_blob = json_blob[first_accolade_index : last_accolade_index + 1].replace(
            '\\"', "'"
        )
        json_data = json.loads(json_blob, strict=False)
        return json_data
    except json.JSONDecodeError as e:
        place = e.pos
        if json_blob[place - 1 : place + 2] == "},\n":
            raise ValueError(
                "JSON is invalid: you probably tried to provide multiple tool calls in one action. PROVIDE ONLY ONE TOOL CALL."
            )
        raise ValueError(
            f"The JSON blob you used is invalid due to the following error: {e}.\n"
            f"JSON blob was: {json_blob}, decoding failed on that specific part of the blob:\n"
            f"'{json_blob[place-4:place+5]}'."
        )
    except Exception as e:
        raise ValueError(f"Error in parsing the JSON blob: {e}")


def parse_code_blob(code_blob: str) -> str:
    try:
        pattern = r"```(?:py|python)?\n(.*?)\n```"
        match = re.search(pattern, code_blob, re.DOTALL)
        if match is None:
            raise ValueError(
                f"No match ground for regex pattern {pattern} in {code_blob=}."
            )
        return match.group(1).strip()

    except Exception as e:
        raise ValueError(
            f"""
The code blob you used is invalid: due to the following error: {e}
This means that the regex pattern {pattern} was not respected: make sure to include code with the correct pattern, for instance:
Thoughts: Your thoughts
Code:
```py
# Your python code here
```<end_action>"""
        )


def parse_json_tool_call(json_blob: str) -> Tuple[str, Union[str, None]]:
    json_blob = json_blob.replace("```json", "").replace("```", "")
    tool_call = parse_json_blob(json_blob)
    if "action" in tool_call and "action_input" in tool_call:
        return tool_call["action"], tool_call["action_input"]
    elif "action" in tool_call:
        return tool_call["action"], None
    else:
        missing_keys = [
            key for key in ["action", "action_input"] if key not in tool_call
        ]
        error_msg = f"Missing keys: {missing_keys} in blob {tool_call}"
        console.print(f"[bold red]{error_msg}[/bold red]")
        raise ValueError(error_msg)


MAX_LENGTH_TRUNCATE_CONTENT = 20000


def truncate_content(
    content: str, max_length: int = MAX_LENGTH_TRUNCATE_CONTENT
) -> str:
    if len(content) <= max_length:
        return content
    else:
        return (
            content[: MAX_LENGTH_TRUNCATE_CONTENT // 2]
            + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
            + content[-MAX_LENGTH_TRUNCATE_CONTENT // 2 :]
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


__all__ = []
