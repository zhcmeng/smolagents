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
# Agents

<Tip warning={true}>

Transformers Agents is an experimental API which is subject to change at any time. Results returned by the agents
can vary as the APIs or underlying models are prone to change.

</Tip>

To learn more about agents and tools make sure to read the [introductory guide](../index). This page
contains the API docs for the underlying classes.

## Agents

Our agents inherit from [`MultiStepAgent`], which means they can act in multiple steps, each step consisting of one thought, then one tool call and execution. Read more in [this conceptual guide](../conceptual_guides/react).

We provide two types of agents, based on the main [`Agent`] class.
  - [`JsonAgent`] writes its tool calls in JSON.
  - [`CodeAgent`] writes its tool calls in Python code.

### BaseAgent

[[autodoc]] BaseAgent


### React agents

[[autodoc]] MultiStepAgent

[[autodoc]] JsonAgent

[[autodoc]] CodeAgent

### ManagedAgent

[[autodoc]] ManagedAgent

### stream_to_gradio

[[autodoc]] stream_to_gradio


## Engines

You're free to create and use your own engines to be usable by the Agents framework.
These engines have the following specification:
1. Follow the [messages format](../chat_templating.md) for its input (`List[Dict[str, str]]`) and return a string.
2. Stop generating outputs *before* the sequences passed in the argument `stop_sequences`

### TransformersEngine

For convenience, we have added a `TransformersEngine` that implements the points above, taking a pre-initialized `Pipeline` as input.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TransformersEngine

model_name = "HuggingFaceTB/SmolLM-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

engine = TransformersEngine(pipe)
engine([{"role": "user", "content": "Ok!"}], stop_sequences=["great"])
```

[[autodoc]] TransformersEngine

### HfApiEngine

The `HfApiEngine` is an engine that wraps an [HF Inference API](https://huggingface.co/docs/api-inference/index) client for the execution of the LLM.

```python
from transformers import HfApiEngine

messages = [
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "No need to help, take it easy."},
]

HfApiEngine()(messages)
```

[[autodoc]] HfApiEngine
