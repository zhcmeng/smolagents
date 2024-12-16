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
import ast
import base64
import builtins
import importlib
import inspect
import io
import json
import os
import re
import tempfile
import textwrap
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Set

from huggingface_hub import (
    create_repo,
    get_collection,
    hf_hub_download,
    metadata_update,
    upload_folder,
)
from huggingface_hub.utils import RepositoryNotFoundError, build_hf_headers, get_session
from packaging import version

from transformers.dynamic_module_utils import (
    custom_object_save,
    get_class_from_dynamic_module,
    get_imports,
)
from transformers import AutoProcessor
from transformers.utils import (
    TypeHintParsingException,
    cached_file,
    get_json_schema,
    is_accelerate_available,
    is_torch_available,
    is_vision_available,
)
from .types import ImageType, handle_agent_inputs, handle_agent_outputs
from .utils import ImportFinder

import logging

logger = logging.getLogger(__name__)


if is_torch_available():
    import torch

if is_accelerate_available():
    from accelerate import PartialState
    from accelerate.utils import send_to_device


TOOL_CONFIG_FILE = "tool_config.json"


def get_repo_type(repo_id, repo_type=None, **hub_kwargs):
    if repo_type is not None:
        return repo_type
    try:
        hf_hub_download(repo_id, TOOL_CONFIG_FILE, repo_type="space", **hub_kwargs)
        return "space"
    except RepositoryNotFoundError:
        try:
            hf_hub_download(repo_id, TOOL_CONFIG_FILE, repo_type="model", **hub_kwargs)
            return "model"
        except RepositoryNotFoundError:
            raise EnvironmentError(
                f"`{repo_id}` does not seem to be a valid repo identifier on the Hub."
            )
        except Exception:
            return "model"
    except Exception:
        return "space"


def setup_default_tools():
    default_tools = {}
    main_module = importlib.import_module("agents")

    for task_name, tool_class_name in TOOL_MAPPING.items():
        tool_class = getattr(main_module, tool_class_name)
        tool_instance = tool_class()
        default_tools[tool_class.name] = tool_instance

    return default_tools


# docstyle-ignore
APP_FILE_TEMPLATE = """from transformers import launch_gradio_demo
from tool import {class_name}

launch_gradio_demo({class_name})
"""


def validate_after_init(cls, do_validate_forward: bool = True):
    original_init = cls.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.validate_arguments(do_validate_forward=do_validate_forward)

    cls.__init__ = new_init
    return cls

def validate_args_are_self_contained(source_code):
    """Validates that all names in forward method are properly defined.
    In particular it will check that all imports are done within the function."""
    print("CODDDD", source_code)
    tree = ast.parse(textwrap.dedent(source_code))
    
    # Get function arguments
    func_node = tree.body[0]
    arg_names = {arg.arg for arg in func_node.args.args} | {"kwargs"}

    builtin_names = set(vars(builtins))

    class NameChecker(ast.NodeVisitor):
        def __init__(self):
            self.undefined_names = set()
            self.imports = {}
            self.from_imports = {}
            self.assigned_names = set()

        def visit_Import(self, node):
            """Handle simple imports like 'import datetime'."""
            for name in node.names:
                actual_name = name.asname or name.name
                self.imports[actual_name] = (name.name, actual_name)
                
        def visit_ImportFrom(self, node):
            """Handle from imports like 'from datetime import datetime'."""
            module = node.module or ''
            for name in node.names:
                actual_name = name.asname or name.name
                self.from_imports[actual_name] = (module, name.name, actual_name)

        def visit_Assign(self, node):
            """Track variable assignments."""
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.assigned_names.add(target.id)
            self.visit(node.value)
            
        def visit_AnnAssign(self, node):
            """Track annotated assignments."""
            if isinstance(node.target, ast.Name):
                self.assigned_names.add(node.target.id)
            if node.value:
                self.visit(node.value)

        def _handle_for_target(self, target) -> Set[str]:
            """Extract all names from a for loop target."""
            names = set()
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, ast.Tuple):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        names.add(elt.id)
            return names
                
        def visit_For(self, node):
            """Track for-loop target variables and handle enumerate specially."""
            # Add names from the target
            target_names = self._handle_for_target(node.target)
            self.assigned_names.update(target_names)
            
            # Special handling for enumerate
            if (isinstance(node.iter, ast.Call) and 
                isinstance(node.iter.func, ast.Name) and 
                node.iter.func.id == 'enumerate'):
                # For enumerate, if we have "for i, x in enumerate(...)", 
                # both i and x should be marked as assigned
                if isinstance(node.target, ast.Tuple):
                    for elt in node.target.elts:
                        if isinstance(elt, ast.Name):
                            self.assigned_names.add(elt.id)
                            
            # Visit the rest of the node
            self.generic_visit(node)

        def visit_Name(self, node):
            if (isinstance(node.ctx, ast.Load) and not (
                node.id == "tool" or
                node.id in builtin_names or
                node.id in arg_names or 
                node.id == 'self' or
                node.id in self.assigned_names
            )):
                if node.id not in self.from_imports and node.id not in self.imports:
                    self.undefined_names.add(node.id)
                    
        def visit_Attribute(self, node):
            # Skip self.something
            if not (isinstance(node.value, ast.Name) and node.value.id == 'self'):
                self.generic_visit(node)
    
    checker = NameChecker()
    checker.visit(tree)
    
    if checker.undefined_names:
        raise ValueError(
            f"""The following names in forward method are not defined: {', '.join(checker.undefined_names)}.
            Make sure all imports and variables are self-contained within the method.            
            """
        )

AUTHORIZED_TYPES = [
    "string",
    "boolean",
    "integer",
    "number",
    "image",
    "audio",
    "any",
]

CONVERSION_DICT = {"str": "string", "int": "integer", "float": "number"}


class Tool:
    """
    A base class for the functions used by the agent. Subclass this and implement the `forward` method as well as the
    following class attributes:

    - **description** (`str`) -- A short description of what your tool does, the inputs it expects and the output(s) it
      will return. For instance 'This is a tool that downloads a file from a `url`. It takes the `url` as input, and
      returns the text contained in the file'.
    - **name** (`str`) -- A performative name that will be used for your tool in the prompt to the agent. For instance
      `"text-classifier"` or `"image_generator"`.
    - **inputs** (`Dict[str, Dict[str, Union[str, type]]]`) -- The dict of modalities expected for the inputs.
      It has one `type`key and a `description`key.
      This is used by `launch_gradio_demo` or to make a nice space from your tool, and also can be used in the generated
      description for your tool.
    - **output_type** (`type`) -- The type of the tool output. This is used by `launch_gradio_demo`
      or to make a nice space from your tool, and also can be used in the generated description for your tool.

    You can also override the method [`~Tool.setup`] if your tool has an expensive operation to perform before being
    usable (such as loading a model). [`~Tool.setup`] will be called the first time you use your tool, but not at
    instantiation.
    """

    name: str
    description: str
    inputs: Dict[str, Dict[str, Union[str, type]]]
    output_type: str

    def __init__(self, *args, **kwargs):
        self.is_initialized = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        validate_after_init(cls, do_validate_forward=False)


    def validate_arguments(self, do_validate_forward: bool = True):
        required_attributes = {
            "description": str,
            "name": str,
            "inputs": dict,
            "output_type": str,
        }

        for attr, expected_type in required_attributes.items():
            attr_value = getattr(self, attr, None)
            if attr_value is None:
                raise TypeError(f"You must set an attribute {attr}.")
            if not isinstance(attr_value, expected_type):
                raise TypeError(
                    f"Attribute {attr} should have type {expected_type.__name__}, got {type(attr_value)} instead."
                )
        for input_name, input_content in self.inputs.items():
            assert isinstance(
                input_content, dict
            ), f"Input '{input_name}' should be a dictionary."
            assert (
                "type" in input_content and "description" in input_content
            ), f"Input '{input_name}' should have keys 'type' and 'description', has only {list(input_content.keys())}."
            if input_content["type"] not in AUTHORIZED_TYPES:
                raise Exception(
                    f"Input '{input_name}': type '{input_content['type']}' is not an authorized value, should be one of {AUTHORIZED_TYPES}."
                )

        assert getattr(self, "output_type", None) in AUTHORIZED_TYPES
        if do_validate_forward:
            signature = inspect.signature(self.forward)
            if not set(signature.parameters.keys()) == set(self.inputs.keys()):
                raise Exception(
                    "Tool's 'forward' method should take 'self' as its first argument, then its next arguments should match the keys of tool attribute 'inputs'."
                )

    def forward(self, *args, **kwargs):
        return NotImplementedError("Write this method in your subclass of `Tool`.")

    def __call__(self, *args, **kwargs):
        if not self.is_initialized:
            self.setup()
        args, kwargs = handle_agent_inputs(*args, **kwargs)
        outputs = self.forward(*args, **kwargs)
        return handle_agent_outputs(outputs, self.output_type)

    def setup(self):
        """
        Overwrite this method here for any operation that is expensive and needs to be executed before you start using
        your tool. Such as loading a big model.
        """
        self.is_initialized = True

    def save(self, output_dir):
        """
        Saves the relevant code files for your tool so it can be pushed to the Hub. This will copy the code of your
        tool in `output_dir` as well as autogenerate:

        - an `app.py` file so that your tool can be converted to a space
        - a `requirements.txt` containing the names of the module used by your tool (as detected when inspecting its
          code)

        You should only use this method to save tools that are defined in a separate module (not `__main__`).

        Args:
            output_dir (`str`): The folder in which you want to save your tool.
        """
        os.makedirs(output_dir, exist_ok=True)
        class_name = self.__class__.__name__

        # Save tool file
        forward_source_code = inspect.getsource(self.forward)
        validate_args_are_self_contained(forward_source_code)
        tool_code = textwrap.dedent(f"""
            from agents import Tool

            class {class_name}(Tool):
                name = "{self.name}"
                description = \"\"\"{self.description}\"\"\"
                inputs = {json.dumps(self.inputs, separators=(',', ':'))}
                output_type = "{self.output_type}"
            """
        ).strip()

        def add_self_argument(source_code: str) -> str:
            """Add 'self' as first argument to a function definition if not present."""
            pattern = r'def forward\(((?!self)[^)]*)\)'
            
            def replacement(match):
                args = match.group(1).strip()
                if args:  # If there are other arguments
                    return f'def forward(self, {args})'
                return 'def forward(self)'
                
            return re.sub(pattern, replacement, source_code)

        forward_source_code = forward_source_code.replace(self.name, "forward")
        forward_source_code = add_self_argument(forward_source_code)
        forward_source_code = forward_source_code.replace("@tool", "").strip()
        tool_code += "\n\n" + textwrap.indent(forward_source_code, "    ")
        with open(os.path.join(output_dir, "tool.py"), "w", encoding="utf-8") as f:
            f.write(tool_code)

        # Save config file
        config_file = os.path.join(output_dir, "tool_config.json")
        tool_config = {
            "tool_class": self.__class__.__name__,
            "description": self.description,
            "name": self.name,
            "inputs": self.inputs,
            "output_type": str(self.output_type),
        }
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(tool_config, indent=2, sort_keys=True) + "\n")

        # Save app file
        app_file = os.path.join(output_dir, "app.py")
        with open(app_file, "w", encoding="utf-8") as f:
            f.write(
                APP_FILE_TEMPLATE.format(
                    class_name=class_name
                )
            )

        # Save requirements file
        requirements_file = os.path.join(output_dir, "requirements.txt")

        tree = ast.parse(forward_source_code)
        import_finder = ImportFinder()
        import_finder.visit(tree)

        imports = list(set(import_finder.packages))
        with open(requirements_file, "w", encoding="utf-8") as f:
            f.write("agents_package\n" + "\n".join(imports) + "\n")

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload tool",
        private: Optional[bool] = None,
        token: Optional[Union[bool, str]] = None,
        create_pr: bool = False,
    ) -> str:
        """
        Upload the tool to the Hub.

        For this method to work properly, your tool must have been defined in a separate module (not `__main__`).
        For instance:
        ```
        from my_tool_module import MyTool
        my_tool = MyTool()
        my_tool.push_to_hub("my-username/my-space")
        ```

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push your tool to. It should contain your organization name when
                pushing to a given organization.
            commit_message (`str`, *optional*, defaults to `"Upload tool"`):
                Message to commit while pushing.
            private (`bool`, *optional*):
                Whether to make the repo private. If `None` (default), the repo will be public unless the organization's default is private. This value is ignored if the repo already exists.
            token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether or not to create a PR with the uploaded files or directly commit.
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
        metadata_update(repo_id, {"tags": ["tool"]}, repo_type="space")

        with tempfile.TemporaryDirectory() as work_dir:
            # Save all files.
            self.save(work_dir)
            print(work_dir)
            with open(work_dir + "/tool.py", "r") as f:
                print('\n'.join(f.readlines()))
            logger.info(
                f"Uploading the following files to {repo_id}: {','.join(os.listdir(work_dir))}"
            )
            return upload_folder(
                repo_id=repo_id,
                commit_message=commit_message,
                folder_path=work_dir,
                token=token,
                create_pr=create_pr,
                repo_type="space",
            )

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        token: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """
        Loads a tool defined on the Hub.

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
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments that will be split in two: all arguments relevant to the Hub (such as
                `cache_dir`, `revision`, `subfolder`) will be used when downloading the files for your tool, and the
                others will be passed along to its init.
        """
        assert trust_remote_code, "Loading a tool from Hub requires to trust remote code. Make sure you've inspected the repo and pass `trust_remote_code=True` to load the tool."

        hub_kwargs_names = [
            "cache_dir",
            "force_download",
            "resume_download",
            "proxies",
            "revision",
            "repo_type",
            "subfolder",
            "local_files_only",
        ]
        hub_kwargs = {k: v for k, v in kwargs.items() if k in hub_kwargs_names}

        tool_file = "tool.py"

        # Get the tool's tool.py file.
        hub_kwargs["repo_type"] = get_repo_type(repo_id, **hub_kwargs)
        resolved_tool_file = cached_file(
            repo_id,
            tool_file,
            token=token,
            **hub_kwargs,
            _raise_exceptions_for_gated_repo=False,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
        )
        tool_code = resolved_tool_file is not None
        if resolved_tool_file is None:
            resolved_tool_file = cached_file(
                repo_id,
                tool_file,
                token=token,
                **hub_kwargs,
                _raise_exceptions_for_gated_repo=False,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )
        if resolved_tool_file is None:
            raise EnvironmentError(
                f"{repo_id} does not appear to provide a valid configuration in `tool_config.json` or `config.json`."
            )

        with open(resolved_tool_file, encoding="utf-8") as reader:
            tool_code = "".join(reader.readlines())    
        
        # Find the Tool subclass in the namespace
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the code to a file
            module_path = os.path.join(temp_dir, "tool.py")
            with open(module_path, "w") as f:
                f.write(tool_code)

            print("TOOLCODE:\n", tool_code)

            # Load module from file path
            spec = importlib.util.spec_from_file_location("custom_tool", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find and instantiate the Tool class
            for item_name in dir(module):
                item = getattr(module, item_name)
                if isinstance(item, type) and issubclass(item, Tool) and item != Tool:
                    tool_class = item
                    break

            if tool_class is None:
                raise ValueError("No Tool subclass found in the code.")
        
        if not isinstance(tool_class.inputs, dict):
            tool_class.inputs = ast.literal_eval(tool_class.inputs)

        return tool_class(**kwargs)


    @staticmethod
    def from_space(
        space_id: str,
        name: str,
        description: str,
        api_name: Optional[str] = None,
        token: Optional[str] = None,
    ):
        """
        Creates a [`Tool`] from a Space given its id on the Hub.

        Args:
            space_id (`str`):
                The id of the Space on the Hub.
            name (`str`):
                The name of the tool.
            description (`str`):
                The description of the tool.
            api_name (`str`, *optional*):
                The specific api_name to use, if the space has several tabs. If not precised, will default to the first available api.
            token (`str`, *optional*):
                Add your token to access private spaces or increase your GPU quotas.
        Returns:
            [`Tool`]:
                The Space, as a tool.

        Examples:
        ```
        image_generator = Tool.from_space(
            space_id="black-forest-labs/FLUX.1-schnell",
            name="image-generator",
            description="Generate an image from a prompt"
        )
        image = image_generator("Generate an image of a cool surfer in Tahiti")
        ```
        ```
        face_swapper = Tool.from_space(
            "tuan2308/face-swap",
            "face_swapper",
            "Tool that puts the face shown on the first image on the second image. You can give it paths to images.",
        )
        image = face_swapper('./aymeric.jpeg', './ruth.jpg')
        ```
        """
        from gradio_client import Client, handle_file
        from gradio_client.utils import is_http_url_like

        class SpaceToolWrapper(Tool):
            def __init__(
                self,
                space_id: str,
                name: str,
                description: str,
                api_name: Optional[str] = None,
                token: Optional[str] = None,
            ):
                self.name = name
                self.description = description
                self.client = Client(space_id, hf_token=token)
                space_description = self.client.view_api(
                    return_format="dict", print_info=False
                )["named_endpoints"]

                # If api_name is not defined, take the first of the available APIs for this space
                if api_name is None:
                    api_name = list(space_description.keys())[0]
                    logger.warning(
                        f"Since `api_name` was not defined, it was automatically set to the first avilable API: `{api_name}`."
                    )
                self.api_name = api_name

                try:
                    space_description_api = space_description[api_name]
                except KeyError:
                    raise KeyError(
                        f"Could not find specified {api_name=} among available api names."
                    )

                self.inputs = {}
                for parameter in space_description_api["parameters"]:
                    if not parameter["parameter_has_default"]:
                        parameter_type = parameter["type"]["type"]
                        if parameter_type == "object":
                            parameter_type = "any"
                        self.inputs[parameter["parameter_name"]] = {
                            "type": parameter_type,
                            "description": parameter["python_type"]["description"],
                        }
                output_component = space_description_api["returns"][0]["component"]
                if output_component == "Image":
                    self.output_type = "image"
                elif output_component == "Audio":
                    self.output_type = "audio"
                else:
                    self.output_type = "any"
                self.is_initialized = True

            def sanitize_argument_for_prediction(self, arg):
                if isinstance(arg, ImageType):
                    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    arg.save(temp_file.name)
                    arg = temp_file.name
                if (
                    isinstance(arg, (str, Path))
                    and Path(arg).exists()
                    and Path(arg).is_file()
                ) or is_http_url_like(arg):
                    arg = handle_file(arg)
                return arg

            def forward(self, *args, **kwargs):
                # Preprocess args and kwargs:
                args = list(args)
                for i, arg in enumerate(args):
                    args[i] = self.sanitize_argument_for_prediction(arg)
                for arg_name, arg in kwargs.items():
                    kwargs[arg_name] = self.sanitize_argument_for_prediction(arg)

                output = self.client.predict(*args, api_name=self.api_name, **kwargs)
                if isinstance(output, tuple) or isinstance(output, list):
                    return output[
                        0
                    ]  # Sometime the space also returns the generation seed, in which case the result is at index 0
                return output

        return SpaceToolWrapper(
            space_id=space_id, name=name, description=description, api_name=api_name, token=token
        )

    @staticmethod
    def from_gradio(gradio_tool):
        """
        Creates a [`Tool`] from a gradio tool.
        """
        import inspect

        class GradioToolWrapper(Tool):
            def __init__(self, _gradio_tool):
                self.name = _gradio_tool.name
                self.description = _gradio_tool.description
                self.output_type = "string"
                self._gradio_tool = _gradio_tool
                func_args = list(inspect.signature(_gradio_tool.run).parameters.items())
                self.inputs = {
                    key: {"type": CONVERSION_DICT[value.annotation], "description": ""}
                    for key, value in func_args
                }
                self.forward = self._gradio_tool.run

        return GradioToolWrapper(gradio_tool)

    @staticmethod
    def from_langchain(langchain_tool):
        """
        Creates a [`Tool`] from a langchain tool.
        """

        class LangChainToolWrapper(Tool):
            def __init__(self, _langchain_tool):
                self.name = _langchain_tool.name.lower()
                self.description = _langchain_tool.description
                self.inputs = _langchain_tool.args.copy()
                for input_content in self.inputs.values():
                    if "title" in input_content:
                        input_content.pop("title")
                    input_content["description"] = ""
                self.output_type = "string"
                self.langchain_tool = _langchain_tool

            def forward(self, *args, **kwargs):
                tool_input = kwargs.copy()
                for index, argument in enumerate(args):
                    if index < len(self.inputs):
                        input_key = next(iter(self.inputs))
                        tool_input[input_key] = argument
                return self.langchain_tool.run(tool_input)

        return LangChainToolWrapper(langchain_tool)


DEFAULT_TOOL_DESCRIPTION_TEMPLATE = """
- {{ tool.name }}: {{ tool.description }}
    Takes inputs: {{tool.inputs}}
    Returns an output of type: {{tool.output_type}}
"""


def get_tool_description_with_args(
    tool: Tool, description_template: Optional[str] = None
) -> str:
    if description_template is None:
        description_template = DEFAULT_TOOL_DESCRIPTION_TEMPLATE
    compiled_template = compile_jinja_template(description_template)
    rendered = compiled_template.render(
        tool=tool,
    )
    return rendered


@lru_cache
def compile_jinja_template(template):
    try:
        import jinja2
        from jinja2.exceptions import TemplateError
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError:
        raise ImportError("template requires jinja2 to be installed.")

    if version.parse(jinja2.__version__) < version.parse("3.1.0"):
        raise ImportError(
            "template requires jinja2>=3.1.0 to be installed. Your version is "
            f"{jinja2.__version__}."
        )

    def raise_exception(message):
        raise TemplateError(message)

    jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    jinja_env.globals["raise_exception"] = raise_exception
    return jinja_env.from_string(template)


def launch_gradio_demo(tool_class: Tool):
    """
    Launches a gradio demo for a tool. The corresponding tool class needs to properly implement the class attributes
    `inputs` and `output_type`.

    Args:
        tool_class (`type`): The class of the tool for which to launch the demo.
    """
    try:
        import gradio as gr
    except ImportError:
        raise ImportError(
            "Gradio should be installed in order to launch a gradio demo."
        )

    tool = tool_class()

    def fn(*args, **kwargs):
        return tool(*args, **kwargs)

    TYPE_TO_COMPONENT_CLASS_MAPPING = {
        "image": gr.Image,
        "audio": gr.Audio,
        "string": gr.Textbox,
        "integer": gr.Textbox,
        "number": gr.Textbox,
    }

    gradio_inputs = []
    for input_name, input_details in tool_class.inputs.items():
        input_gradio_component_class = TYPE_TO_COMPONENT_CLASS_MAPPING[
            input_details["type"]
        ]
        new_component = input_gradio_component_class(label=input_name)
        gradio_inputs.append(new_component)

    output_gradio_componentclass = TYPE_TO_COMPONENT_CLASS_MAPPING[
        tool_class.output_type
    ]
    gradio_output = output_gradio_componentclass(label=input_name)

    gr.Interface(
        fn=fn,
        inputs=gradio_inputs,
        outputs=gradio_output,
        title=tool_class.__name__,
        article=tool.description,
    ).launch()


TOOL_MAPPING = {
    "python_interpreter": "PythonInterpreterTool",
    "web_search": "DuckDuckGoSearchTool",
}


def load_tool(
        task_or_repo_id,
        model_repo_id: Optional[str] = None,
        token: Optional[str] = None,
        trust_remote_code: bool=False,
        **kwargs
    ):
    """
    Main function to quickly load a tool, be it on the Hub or in the Transformers library.

    <Tip warning={true}>

    Loading a tool means that you'll download the tool and execute it locally.
    ALWAYS inspect the tool you're downloading before loading it within your runtime, as you would do when
    installing a package using pip/npm/apt.

    </Tip>

    Args:
        task_or_repo_id (`str`):
            The task for which to load the tool or a repo ID of a tool on the Hub. Tasks implemented in Transformers
            are:

            - `"document_question_answering"`
            - `"image_question_answering"`
            - `"speech_to_text"`
            - `"text_to_speech"`
            - `"translation"`

        model_repo_id (`str`, *optional*):
            Use this argument to use a different model than the default one for the tool you selected.
        token (`str`, *optional*):
            The token to identify you on hf.co. If unset, will use the token generated when running `huggingface-cli
            login` (stored in `~/.huggingface`).
        trust_remote_code (`bool`, *optional*, defaults to False):
            This needs to be accepted in order to load a tool from Hub.
        kwargs (additional keyword arguments, *optional*):
            Additional keyword arguments that will be split in two: all arguments relevant to the Hub (such as
            `cache_dir`, `revision`, `subfolder`) will be used when downloading the files for your tool, and the others
            will be passed along to its init.
    """
    if task_or_repo_id in TOOL_MAPPING:
        tool_class_name = TOOL_MAPPING[task_or_repo_id]
        main_module = importlib.import_module("agents")
        tools_module = main_module
        tool_class = getattr(tools_module, tool_class_name)
        return tool_class(model_repo_id, token=token, **kwargs)
    else:
        logger.warning_once(
            f"You're loading a tool from the Hub from {model_repo_id}. Please make sure this is a source that you "
            f"trust as the code within that tool will be executed on your machine. Always verify the code of "
            f"the tools that you load. We recommend specifying a `revision` to ensure you're loading the "
            f"code that you have checked."
        )
        return Tool.from_hub(
            task_or_repo_id, model_repo_id=model_repo_id, token=token, trust_remote_code=trust_remote_code, **kwargs
        )


def add_description(description):
    """
    A decorator that adds a description to a function.
    """

    def inner(func):
        func.description = description
        func.name = func.__name__
        return func

    return inner


## Will move to the Hub
class EndpointClient:
    def __init__(self, endpoint_url: str, token: Optional[str] = None):
        self.headers = {
            **build_hf_headers(token=token),
            "Content-Type": "application/json",
        }
        self.endpoint_url = endpoint_url

    @staticmethod
    def encode_image(image):
        _bytes = io.BytesIO()
        image.save(_bytes, format="PNG")
        b64 = base64.b64encode(_bytes.getvalue())
        return b64.decode("utf-8")

    @staticmethod
    def decode_image(raw_image):
        if not is_vision_available():
            raise ImportError(
                "This tool returned an image but Pillow is not installed. Please install it (`pip install Pillow`)."
            )

        from PIL import Image

        b64 = base64.b64decode(raw_image)
        _bytes = io.BytesIO(b64)
        return Image.open(_bytes)

    def __call__(
        self,
        inputs: Optional[Union[str, Dict, List[str], List[List[str]]]] = None,
        params: Optional[Dict] = None,
        data: Optional[bytes] = None,
        output_image: bool = False,
    ) -> Any:
        # Build payload
        payload = {}
        if inputs:
            payload["inputs"] = inputs
        if params:
            payload["parameters"] = params

        # Make API call
        response = get_session().post(
            self.endpoint_url, headers=self.headers, json=payload, data=data
        )

        # By default, parse the response for the user.
        if output_image:
            return self.decode_image(response.content)
        else:
            return response.json()


class ToolCollection:
    """
    Tool collections enable loading all Spaces from a collection in order to be added to the agent's toolbox.

    > [!NOTE]
    > Only Spaces will be fetched, so you can feel free to add models and datasets to your collection if you'd
    > like for this collection to showcase them.

    Args:
        collection_slug (str):
            The collection slug referencing the collection.
        token (str, *optional*):
            The authentication token if the collection is private.

    Example:

    ```py
    >>> from transformers import ToolCollection, CodeAgent

    >>> image_tool_collection = ToolCollection(collection_slug="huggingface-tools/diffusion-tools-6630bb19a942c2306a2cdb6f")
    >>> agent = CodeAgent(tools=[*image_tool_collection.tools], add_base_tools=True)

    >>> agent.run("Please draw me a picture of rivers and lakes.")
    ```
    """

    def __init__(self, collection_slug: str, token: Optional[str] = None):
        self._collection = get_collection(collection_slug, token=token)
        self._hub_repo_ids = {
            item.item_id for item in self._collection.items if item.item_type == "space"
        }
        self.tools = {Tool.from_hub(repo_id) for repo_id in self._hub_repo_ids}


def tool(tool_function: Callable) -> Tool:
    """
    Converts a function into an instance of a Tool subclass.

    Args:
        tool_function: Your function. Should have type hints for each input and a type hint for the output.
        Should also have a docstring description including an 'Args:' part where each argument is described.
    """
    parameters = get_json_schema(tool_function)["function"]
    if "return" not in parameters:
        raise TypeHintParsingException(
            "Tool return type not found: make sure your function has a return type hint!"
        )
    class_name = ''.join([el.title() for el in parameters['name'].split('_')])

    if parameters["return"]["type"] == "object":
        parameters["return"]["type"] = "any"

    class SpecificTool(Tool):
        name = parameters["name"]
        description = parameters["description"]
        inputs = parameters["parameters"]["properties"]
        output_type = parameters["return"]["type"]

        @wraps(tool_function)
        def forward(self, *args, **kwargs):
            return tool_function(*args, **kwargs)

    original_signature = inspect.signature(tool_function)
    new_parameters = [
        inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ] + list(original_signature.parameters.values())
    new_signature = original_signature.replace(parameters=new_parameters)
    SpecificTool.forward.__signature__ = new_signature
    SpecificTool.__name__ = class_name
    return SpecificTool()


HUGGINGFACE_DEFAULT_TOOLS = {}


class Toolbox:
    """
    The toolbox contains all tools that the agent can perform operations with, as well as a few methods to
    manage them.

    Args:
        tools (`List[Tool]`):
            The list of tools to instantiate the toolbox with
        add_base_tools (`bool`, defaults to `False`, *optional*, defaults to `False`):
            Whether to add the tools available within `transformers` to the toolbox.
    """

    def __init__(self, tools: List[Tool], add_base_tools: bool = False):
        self._tools = {tool.name: tool for tool in tools}
        if add_base_tools:
            self.add_base_tools()

    def add_base_tools(self, add_python_interpreter: bool = False):
        global HUGGINGFACE_DEFAULT_TOOLS
        if len(HUGGINGFACE_DEFAULT_TOOLS.keys()) == 0:
            HUGGINGFACE_DEFAULT_TOOLS = setup_default_tools()
        for tool in HUGGINGFACE_DEFAULT_TOOLS.values():
            if tool.name != "python_interpreter" or add_python_interpreter:
                self.add_tool(tool)

    @property
    def tools(self) -> Dict[str, Tool]:
        """Get all tools currently in the toolbox"""
        return self._tools

    def show_tool_descriptions(self, tool_description_template: Optional[str] = None) -> str:
        """
        Returns the description of all tools in the toolbox

        Args:
            tool_description_template (`str`, *optional*):
                The template to use to describe the tools. If not provided, the default template will be used.
        """
        return "\n".join(
            [
                get_tool_description_with_args(tool, tool_description_template)
                for tool in self._tools.values()
            ]
        )

    def add_tool(self, tool: Tool):
        """
        Adds a tool to the toolbox

        Args:
            tool (`Tool`):
                The tool to add to the toolbox.
        """
        if tool.name in self._tools:
            raise KeyError(f"Error: tool '{tool.name}' already exists in the toolbox.")
        self._tools[tool.name] = tool

    def remove_tool(self, tool_name: str):
        """
        Removes a tool from the toolbox

        Args:
            tool_name (`str`):
                The tool to remove from the toolbox.
        """
        if tool_name not in self._tools:
            raise KeyError(
                f"Error: tool {tool_name} not found in toolbox for removal, should be instead one of {list(self._tools.keys())}."
            )
        del self._tools[tool_name]

    def update_tool(self, tool: Tool):
        """
        Updates a tool in the toolbox according to its name.

        Args:
            tool (`Tool`):
                The tool to update to the toolbox.
        """
        if tool.name not in self._tools:
            raise KeyError(
                f"Error: tool {tool.name} not found in toolbox for update, should be instead one of {list(self._tools.keys())}."
            )
        self._tools[tool.name] = tool

    def clear_toolbox(self):
        """Clears the toolbox"""
        self._tools = {}

    def __repr__(self):
        toolbox_description = "Toolbox contents:\n"
        for tool in self._tools.values():
            toolbox_description += f"\t{tool.name}: {tool.description}\n"
        return toolbox_description

__all__ = ["AUTHORIZED_TYPES", "Tool", "tool", "load_tool", "launch_gradio_demo", "Toolbox"]
