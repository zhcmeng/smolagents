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
import json
import math
from dataclasses import dataclass
from math import sqrt
from typing import Dict

from huggingface_hub import hf_hub_download, list_spaces

from transformers.utils import is_offline_mode
from .local_python_executor import LIST_SAFE_MODULES, evaluate_python_code
from .tools import TOOL_CONFIG_FILE, Tool


def custom_print(*args):
    return None


BASE_PYTHON_TOOLS = {
    "print": custom_print,
    "isinstance": isinstance,
    "range": range,
    "float": float,
    "int": int,
    "bool": bool,
    "str": str,
    "set": set,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "round": round,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "degrees": math.degrees,
    "radians": math.radians,
    "pow": math.pow,
    "sqrt": sqrt,
    "len": len,
    "sum": sum,
    "max": max,
    "min": min,
    "abs": abs,
    "enumerate": enumerate,
    "zip": zip,
    "reversed": reversed,
    "sorted": sorted,
    "all": all,
    "any": any,
    "map": map,
    "filter": filter,
    "ord": ord,
    "chr": chr,
    "next": next,
    "iter": iter,
    "divmod": divmod,
    "callable": callable,
    "getattr": getattr,
    "hasattr": hasattr,
    "setattr": setattr,
    "issubclass": issubclass,
    "type": type,
}


@dataclass
class PreTool:
    name: str
    inputs: Dict[str, str]
    output_type: type
    task: str
    description: str
    repo_id: str


def get_remote_tools(logger, organization="huggingface-tools"):
    if is_offline_mode():
        logger.info("You are in offline mode, so remote tools are not available.")
        return {}

    spaces = list_spaces(author=organization)
    tools = {}
    for space_info in spaces:
        repo_id = space_info.id
        resolved_config_file = hf_hub_download(
            repo_id, TOOL_CONFIG_FILE, repo_type="space"
        )
        with open(resolved_config_file, encoding="utf-8") as reader:
            config = json.load(reader)
        task = repo_id.split("/")[-1]
        tools[config["name"]] = PreTool(
            task=task,
            description=config["description"],
            repo_id=repo_id,
            name=task,
            inputs=config["inputs"],
            output_type=config["output_type"],
        )

    return tools


class PythonInterpreterTool(Tool):
    name = "python_interpreter"
    description = "This is a tool that evaluates python code. It can be used to perform calculations."
    inputs = {
        "code": {"type": "string", "description": "The python code to run in interpreter"}
    }
    output_type = "string"

    def __init__(self, *args, authorized_imports=None, **kwargs):
        if authorized_imports is None:
            self.authorized_imports = list(set(LIST_SAFE_MODULES))
        else:
            self.authorized_imports = list(
                set(LIST_SAFE_MODULES) | set(authorized_imports)
            )
        self.inputs = {
            "code": {
                "type": "string",
                "description": (
                    "The code snippet to evaluate. All variables used in this snippet must be defined in this same snippet, "
                    f"else you will get an error. This code can only import the following python libraries: {authorized_imports}."
                ),
            }
        }
        super().__init__(*args, **kwargs)

    def forward(self, code: str) -> str:
        output = str(
            evaluate_python_code(
                code,
                static_tools=BASE_PYTHON_TOOLS,
                authorized_imports=self.authorized_imports,
            )
        )
        return output


class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides a final answer to the given problem."
    inputs = {
        "answer": {"type": "any", "description": "The final answer to the problem"}
    }
    output_type = "any"

    def forward(self, answer):
        return answer


class UserInputTool(Tool):
    name = "user_input"
    description = "Asks for user's input on a specific question"
    inputs = {
        "question": {"type": "string", "description": "The question to ask the user"}
    }
    output_type = "string"

    def forward(self, question):
        user_input = input(f"{question} => ")
        return user_input

__all__ = ["PythonInterpreterTool", "FinalAnswerTool", "UserInputTool"]