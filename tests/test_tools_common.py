# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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
import unittest
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pytest

from transformers import is_torch_available, is_vision_available
from agents.types import (
    AGENT_TYPE_MAPPING,
    AgentAudio,
    AgentImage,
    AgentText,
)
from agents.tools import Tool, tool, AUTHORIZED_TYPES
from transformers.testing_utils import get_tests_dir


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


def create_inputs(tool_inputs: Dict[str, Dict[Union[str, type], str]]):
    inputs = {}

    for input_name, input_desc in tool_inputs.items():
        input_type = input_desc["type"]

        if input_type == "string":
            inputs[input_name] = "Text input"
        elif input_type == "image":
            inputs[input_name] = Image.open(
                Path(get_tests_dir("fixtures")) / "000000039769.png"
            ).resize((512, 512))
        elif input_type == "audio":
            inputs[input_name] = np.ones(3000)
        else:
            raise ValueError(f"Invalid type requested: {input_type}")

    return inputs


def output_type(output):
    if isinstance(output, (str, AgentText)):
        return "string"
    elif isinstance(output, (Image.Image, AgentImage)):
        return "image"
    elif isinstance(output, (torch.Tensor, AgentAudio)):
        return "audio"
    else:
        raise TypeError(f"Invalid output: {output}")


class ToolTesterMixin:
    def test_inputs_output(self):
        self.assertTrue(hasattr(self.tool, "inputs"))
        self.assertTrue(hasattr(self.tool, "output_type"))

        inputs = self.tool.inputs
        self.assertTrue(isinstance(inputs, dict))

        for _, input_spec in inputs.items():
            self.assertTrue("type" in input_spec)
            self.assertTrue("description" in input_spec)
            self.assertTrue(input_spec["type"] in AUTHORIZED_TYPES)
            self.assertTrue(isinstance(input_spec["description"], str))

        output_type = self.tool.output_type
        self.assertTrue(output_type in AUTHORIZED_TYPES)

    def test_common_attributes(self):
        self.assertTrue(hasattr(self.tool, "description"))
        self.assertTrue(hasattr(self.tool, "name"))
        self.assertTrue(hasattr(self.tool, "inputs"))
        self.assertTrue(hasattr(self.tool, "output_type"))

    def test_agent_type_output(self):
        inputs = create_inputs(self.tool.inputs)
        output = self.tool(**inputs)
        if self.tool.output_type != "any":
            agent_type = AGENT_TYPE_MAPPING[self.tool.output_type]
            self.assertTrue(isinstance(output, agent_type))


class ToolTests(unittest.TestCase):
    def test_tool_init_with_decorator(self):
        @tool
        def coolfunc(a: str, b: int) -> float:
            """Cool function

            Args:
                a: The first argument
                b: The second one
            """
            return b + 2, a

        assert coolfunc.output_type == "number"

    def test_tool_init_vanilla(self):
        class HFModelDownloadsTool(Tool):
            name = "model_download_counter"
            description = """
            This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub.
            It returns the name of the checkpoint."""

            inputs = {
                "task": {
                    "type": "string",
                    "description": "the task category (such as text-classification, depth-estimation, etc)",
                }
            }
            output_type = "integer"

            def forward(self, task):
                return "best model"

        tool = HFModelDownloadsTool()
        assert list(tool.inputs.keys())[0] == "task"

    def test_tool_init_decorator_raises_issues(self):
        with pytest.raises(Exception) as e:

            @tool
            def coolfunc(a: str, b: int):
                """Cool function

                Args:
                    a: The first argument
                    b: The second one
                """
                return a + b

            assert coolfunc.output_type == "number"
        assert "Tool return type not found" in str(e)

        with pytest.raises(Exception) as e:

            @tool
            def coolfunc(a: str, b: int) -> int:
                """Cool function

                Args:
                    a: The first argument
                """
                return b + a

            assert coolfunc.output_type == "number"
        assert "docstring has no description for the argument" in str(e)

    def test_tool_definition_raises_error_imports_outside_function(self):
        with pytest.raises(Exception) as e:
            from datetime import datetime

            @tool
            def get_current_time() -> str:
                """
                Gets the current time.
                """
                return str(datetime.now())

        assert "datetime" in str(e)

        # Also test with classic definition
        with pytest.raises(Exception) as e:

            class GetCurrentTimeTool(Tool):
                name = "get_current_time_tool"
                description = "Gets the current time"
                inputs = {}
                output_type = "string"

                def forward(self):
                    return str(datetime.now())

        assert "datetime" in str(e)

    def test_tool_definition_raises_no_error_imports_in_function(self):
        @tool
        def get_current_time() -> str:
            """
            Gets the current time.
            """
            from datetime import datetime

            return str(datetime.now())

        class GetCurrentTimeTool(Tool):
            name = "get_current_time_tool"
            description = "Gets the current time"
            inputs = {}
            output_type = "string"

            def forward(self):
                from datetime import datetime
                return str(datetime.now())
