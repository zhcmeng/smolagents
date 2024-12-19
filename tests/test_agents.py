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
import os
import tempfile
import unittest
import uuid
import pytest

from pathlib import Path

from agents.types import AgentText
from agents.agents import (
    AgentMaxIterationsError,
    ManagedAgent,
    CodeAgent,
    JsonAgent,
    Toolbox,
    ToolCall,
)
from agents.tool import tool
from agents.default_tools import PythonInterpreterTool
from transformers.testing_utils import get_tests_dir


def get_new_path(suffix="") -> str:
    directory = tempfile.mkdtemp()
    return os.path.join(directory, str(uuid.uuid4()) + suffix)


def fake_json_llm(messages, stop_sequences=None, grammar=None) -> str:
    prompt = str(messages)

    if "special_marker" not in prompt:
        return """
Thought: I should multiply 2 by 3.6452. special_marker
Action:
{
    "action": "python_interpreter",
    "action_input": {"code": "2*3.6452"}
}
"""
    else:  # We're at step 2
        return """
Thought: I can now answer the initial question
Action:
{
    "action": "final_answer",
    "action_input": {"answer": "7.2904"}
}
"""


def fake_json_llm_image(messages, stop_sequences=None, grammar=None) -> str:
    prompt = str(messages)

    if "special_marker" not in prompt:
        return """
Thought: I should generate an image. special_marker
Action:
{
    "action": "fake_image_generation_tool",
    "action_input": {"prompt": "An image of a cat"}
}
"""
    else:  # We're at step 2
        return """
Thought: I can now answer the initial question
Action:
{
    "action": "final_answer",
    "action_input": "image.png"
}
"""


def fake_code_llm(messages, stop_sequences=None, grammar=None) -> str:
    prompt = str(messages)
    if "special_marker" not in prompt:
        return """
Thought: I should multiply 2 by 3.6452. special_marker
Code:
```py
result = 2**3.6452
```<end_code>
"""
    else:  # We're at step 2
        return """
Thought: I can now answer the initial question
Code:
```py
final_answer(7.2904)
```<end_code>
"""


def fake_code_llm_error(messages, stop_sequences=None) -> str:
    prompt = str(messages)
    if "special_marker" not in prompt:
        return """
Thought: I should multiply 2 by 3.6452. special_marker
Code:
```py
print = 2
```<end_code>
"""
    else:  # We're at step 2
        return """
Thought: I can now answer the initial question
Code:
```py
final_answer("got an error")
```<end_code>
"""


def fake_code_functiondef(messages, stop_sequences=None) -> str:
    prompt = str(messages)
    if "special_marker" not in prompt:
        return """
Thought: Let's define the function. special_marker
Code:
```py
import numpy as np

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
```<end_code>
"""
    else:  # We're at step 2
        return """
Thought: I can now answer the initial question
Code:
```py
x, w = [0, 1, 2, 3, 4, 5], 2
res = moving_average(x, w)
final_answer(res)
```<end_code>
"""


def fake_code_llm_oneshot(messages, stop_sequences=None, grammar=None) -> str:
    return """
Thought: I should multiply 2 by 3.6452. special_marker
Code:
```py
result = python_interpreter(code="2*3.6452")
final_answer(result)
```
"""


def fake_code_llm_no_return(messages, stop_sequences=None, grammar=None) -> str:
    return """
Thought: I should multiply 2 by 3.6452. special_marker
Code:
```py
result = python_interpreter(code="2*3.6452")
print(result)
```
"""


class AgentTests(unittest.TestCase):
    def test_fake_oneshot_code_agent(self):
        agent = CodeAgent(
            tools=[PythonInterpreterTool()], llm_engine=fake_code_llm_oneshot
        )
        output = agent.run("What is 2 multiplied by 3.6452?", oneshot=True)
        assert isinstance(output, str)
        assert output == "7.2904"

    def test_fake_json_agent(self):
        agent = JsonAgent(tools=[PythonInterpreterTool()], llm_engine=fake_json_llm)
        output = agent.run("What is 2 multiplied by 3.6452?")
        assert isinstance(output, str)
        assert output == "7.2904"
        assert agent.logs[1].task == "What is 2 multiplied by 3.6452?"
        assert agent.logs[2].observations == "7.2904"
        assert (
            agent.logs[3].llm_output
            == """
Thought: I can now answer the initial question
Action:
{
    "action": "final_answer",
    "action_input": {"answer": "7.2904"}
}
"""
        )

    def test_json_agent_handles_image_tool_outputs(self):
        from PIL import Image

        @tool
        def fake_image_generation_tool(prompt: str) -> Image.Image:
            """Tool that generates an image.

            Args:
                prompt: The prompt
            """
            return Image.open(Path(get_tests_dir("fixtures")) / "000000039769.png")

        agent = JsonAgent(
            tools=[fake_image_generation_tool], llm_engine=fake_json_llm_image
        )
        output = agent.run("Make me an image.")
        assert isinstance(output, Image.Image)
        assert isinstance(agent.state["image.png"], Image.Image)

    def test_fake_code_agent(self):
        agent = CodeAgent(tools=[PythonInterpreterTool()], llm_engine=fake_code_llm)
        output = agent.run("What is 2 multiplied by 3.6452?")
        assert isinstance(output, float)
        assert output == 7.2904
        assert agent.logs[1].task == "What is 2 multiplied by 3.6452?"
        assert agent.logs[3].tool_call == ToolCall(
            tool_name="python_interpreter",
            tool_arguments="final_answer(7.2904)",
        )

    def test_additional_args_added_to_task(self):
        agent = CodeAgent(tools=[], llm_engine=fake_code_llm)
        output = agent.run("What is 2 multiplied by 3.6452?", additional_instruction="Remember this.")
        assert "Remember this" in agent.task
        assert "Remember this" in str(agent.prompt_messages)

    def test_reset_conversations(self):
        agent = CodeAgent(tools=[PythonInterpreterTool()], llm_engine=fake_code_llm)
        output = agent.run("What is 2 multiplied by 3.6452?", reset=True)
        assert output == 7.2904
        assert len(agent.logs) == 4

        output = agent.run("What is 2 multiplied by 3.6452?", reset=False)
        assert output == 7.2904
        assert len(agent.logs) == 6

        output = agent.run("What is 2 multiplied by 3.6452?", reset=True)
        assert output == 7.2904
        assert len(agent.logs) == 4

    def test_code_agent_code_errors_show_offending_lines(self):
        agent = CodeAgent(
            tools=[PythonInterpreterTool()], llm_engine=fake_code_llm_error
        )
        output = agent.run("What is 2 multiplied by 3.6452?")
        assert isinstance(output, AgentText)
        assert output == "got an error"
        assert "Evaluation stopped at line 'print = 2' because of" in str(agent.logs)

    def test_setup_agent_with_empty_toolbox(self):
        JsonAgent(llm_engine=fake_json_llm, tools=[])

    def test_fails_max_iterations(self):
        agent = CodeAgent(
            tools=[PythonInterpreterTool()],
            llm_engine=fake_code_llm_no_return,  # use this callable because it never ends
            max_iterations=5,
        )
        agent.run("What is 2 multiplied by 3.6452?")
        assert len(agent.logs) == 8
        assert type(agent.logs[-1].error) is AgentMaxIterationsError

    def test_init_agent_with_different_toolsets(self):
        toolset_1 = []
        agent = CodeAgent(tools=toolset_1, llm_engine=fake_code_llm)
        assert (
            len(agent.toolbox.tools) == 1
        )  # when no tools are provided, only the final_answer tool is added by default

        toolset_2 = [PythonInterpreterTool(), PythonInterpreterTool()]
        agent = CodeAgent(tools=toolset_2, llm_engine=fake_code_llm)
        assert (
            len(agent.toolbox.tools) == 2
        )  # deduplication of tools, so only one python_interpreter tool is added in addition to final_answer

        toolset_3 = Toolbox(toolset_2)
        agent = CodeAgent(tools=toolset_3, llm_engine=fake_code_llm)
        assert (
            len(agent.toolbox.tools) == 2
        )  # same as previous one, where toolset_3 is an instantiation of previous one

        # check that add_base_tools will not interfere with existing tools
        with pytest.raises(KeyError) as e:
            agent = JsonAgent(
                tools=toolset_3, llm_engine=fake_json_llm, add_base_tools=True
            )
        assert "already exists in the toolbox" in str(e)

        # check that python_interpreter base tool does not get added to code agents
        agent = CodeAgent(tools=[], llm_engine=fake_code_llm, add_base_tools=True)
        assert len(agent.toolbox.tools) == 2  # added final_answer tool + search

    def test_function_persistence_across_steps(self):
        agent = CodeAgent(
            tools=[],
            llm_engine=fake_code_functiondef,
            max_iterations=2,
            additional_authorized_imports=["numpy"],
        )
        res = agent.run("ok")
        assert res[0] == 0.5

    def test_init_managed_agent(self):
        agent = CodeAgent(tools=[], llm_engine=fake_code_functiondef)
        managed_agent = ManagedAgent(agent, name="managed_agent", description="Empty")
        assert managed_agent.name == "managed_agent"
        assert managed_agent.description == "Empty"

    def test_agent_description_gets_correctly_inserted_in_system_prompt(self):
        agent = CodeAgent(tools=[], llm_engine=fake_code_functiondef)
        managed_agent = ManagedAgent(agent, name="managed_agent", description="Empty")
        manager_agent = CodeAgent(
            tools=[],
            llm_engine=fake_code_functiondef,
            managed_agents=[managed_agent],
        )
        assert "You can also give requests to team members." not in agent.system_prompt
        print("ok1")
        assert "{{managed_agents_descriptions}}" not in agent.system_prompt
        assert (
            "You can also give requests to team members." in manager_agent.system_prompt
        )
