<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
# Tools

[[open-in-colab]]

Here, we're going to see advanced tool usage.

> [!TIP]
> If you're new to `agents`, make sure to first read the main [agents documentation](./agents).


### Directly define a tool by subclassing Tool, and share it to the Hub

Let's take again the tool example from main documentation, for which we had implemented a `tool` decorator.

If you need to add variation, like custom attributes for your tool, you can build your tool following the fine-grained method: building a class that inherits from the [`Tool`] superclass.

The custom tool needs:
- An attribute `name`, which corresponds to the name of the tool itself. The name usually describes what the tool does. Since the code returns the model with the most downloads for a task, let's name it `model_download_counter`.
- An attribute `description` is used to populate the agent's system prompt.
- An `inputs` attribute, which is a dictionary with keys `"type"` and `"description"`. It contains information that helps the Python interpreter make educated choices about the input.
- An `output_type` attribute, which specifies the output type.
- A `forward` method which contains the inference code to be executed.

The types for both `inputs` and `output_type` should be amongst [Pydantic formats](https://docs.pydantic.dev/latest/concepts/json_schema/#generating-json-schema), they can be either of these: [`~AUTHORIZED_TYPES`].


```python
from transformers import Tool
from huggingface_hub import list_models

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
    output_type = "string"

    def forward(self, task: str):
        model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
        return model.id
```

Now that the custom `HfModelDownloadsTool` class is ready, you can save it to a file named `model_downloads.py` and import it for use.


```python
from model_downloads import HFModelDownloadsTool

tool = HFModelDownloadsTool()
```

You can also share your custom tool to the Hub by calling [`~Tool.push_to_hub`] on the tool. Make sure you've created a repository for it on the Hub and are using a token with read access.

```python
tool.push_to_hub("{your_username}/hf-model-downloads")
```

Load the tool with the [`~Tool.load_tool`] function and pass it to the `tools` parameter in your agent.

```python
from transformers import load_tool, CodeAgent

model_download_tool = load_tool("m-ric/hf-model-downloads")
```

### Import a Space as a tool 🚀

You can directly import a Space from the Hub as a tool using the [`Tool.from_space`] method!

You only need to provide the id of the Space on the Hub, its name, and a description that will help you agent understand what the tool does. Under the hood, this will use [`gradio-client`](https://pypi.org/project/gradio-client/) library to call the Space.

For instance, let's import the [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) Space from the Hub and use it to generate an image.

```python
from transformers import Tool

image_generation_tool = Tool.from_space(
    "black-forest-labs/FLUX.1-dev",
    name="image_generator",
    description="Generate an image from a prompt")

image_generation_tool("A sunny beach")
```
And voilà, here's your image! 🏖️

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/sunny_beach.webp">

Then you can use this tool just like any other tool.  For example, let's improve the prompt  `a rabbit wearing a space suit` and generate an image of it.

```python
from transformers import CodeAgent

agent = CodeAgent(tools=[image_generation_tool])

agent.run(
    "Improve this prompt, then generate an image of it.", prompt='A rabbit wearing a space suit'
)
```

```text
=== Agent thoughts:
improved_prompt could be "A bright blue space suit wearing rabbit, on the surface of the moon, under a bright orange sunset, with the Earth visible in the background"

Now that I have improved the prompt, I can use the image generator tool to generate an image based on this prompt.
>>> Agent is executing the code below:
image = image_generator(prompt="A bright blue space suit wearing rabbit, on the surface of the moon, under a bright orange sunset, with the Earth visible in the background")
final_answer(image)
```

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rabbit_spacesuit_flux.webp">

How cool is this? 🤩

### Use gradio-tools

[gradio-tools](https://github.com/freddyaboulton/gradio-tools) is a powerful library that allows using Hugging
Face Spaces as tools. It supports many existing Spaces as well as custom Spaces.

Transformers supports `gradio_tools` with the [`Tool.from_gradio`] method. For example, let's use the [`StableDiffusionPromptGeneratorTool`](https://github.com/freddyaboulton/gradio-tools/blob/main/gradio_tools/tools/prompt_generator.py) from `gradio-tools` toolkit for improving prompts to generate better images.

Import and instantiate the tool, then pass it to the `Tool.from_gradio` method:

```python
from gradio_tools import StableDiffusionPromptGeneratorTool
from agents import Tool, load_tool, CodeAgent

gradio_prompt_generator_tool = StableDiffusionPromptGeneratorTool()
prompt_generator_tool = Tool.from_gradio(gradio_prompt_generator_tool)
```

> [!WARNING]
> gradio-tools require *textual* inputs and outputs even when working with different modalities like image and audio objects. Image and audio inputs and outputs are currently incompatible.

### Use LangChain tools

We love Langchain and think it has a very compelling suite of tools.
To import a tool from LangChain, use the `from_langchain()` method.

Here is how you can use it to recreate the intro's search result using a LangChain web search tool.
This tool will need `pip install google-search-results` to work properly.
```python
from langchain.agents import load_tools
from agents import Tool, CodeAgent

search_tool = Tool.from_langchain(load_tools(["serpapi"])[0])

agent = CodeAgent(tools=[search_tool])

agent.run("How many more blocks (also denoted as layers) are in BERT base encoder compared to the encoder from the architecture proposed in Attention is All You Need?")
```

### Manage your agent's toolbox

You can manage an agent's toolbox by adding or replacing a tool.

Let's add the `model_download_tool` to an existing agent initialized with only the default toolbox.

```python
from transformers import CodeAgent

agent = CodeAgent(tools=[], llm_engine=llm_engine, add_base_tools=True)
agent.toolbox.add_tool(model_download_tool)
```
Now we can leverage both the new tool and the previous text-to-speech tool:

```python
agent.run(
    "Can you read out loud the name of the model that has the most downloads in the 'text-to-video' task on the Hugging Face Hub and return the audio?"
)
```


| **Audio**                                                                                                                                            |
|------------------------------------------------------------------------------------------------------------------------------------------------------|
| <audio controls><source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/damo.wav" type="audio/wav"/> |


> [!WARNING]
> Beware when adding tools to an agent that already works well because it can bias selection towards your tool or select another tool other than the one already defined.


Use the `agent.toolbox.update_tool()` method to replace an existing tool in the agent's toolbox.
This is useful if your new tool is a one-to-one replacement of the existing tool because the agent already knows how to perform that specific task.
Just make sure the new tool follows the same API as the replaced tool or adapt the system prompt template to ensure all examples using the replaced tool are updated.


### Use a collection of tools

You can leverage tool collections by using the ToolCollection object, with the slug of the collection you want to use.
Then pass them as a list to initialize you agent, and start using them!

```py
from transformers import ToolCollection, CodeAgent

image_tool_collection = ToolCollection(collection_slug="huggingface-tools/diffusion-tools-6630bb19a942c2306a2cdb6f")
agent = CodeAgent(tools=[*image_tool_collection.tools], add_base_tools=True)

agent.run("Please draw me a picture of rivers and lakes.")
```

To speed up the start, tools are loaded only if called by the agent.

This gets you this image:

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/rivers_and_lakes.png">
