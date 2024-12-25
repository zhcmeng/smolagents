<!---
Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<p align="center">
    <!-- Uncomment when CircleCI is set up
    <a href="https://circleci.com/gh/huggingface/accelerate"><img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/master"></a>
    -->
    <a href="https://github.com/huggingface/agents/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/huggingface/agents.svg?color=blue"></a>
    <a href="https://huggingface.co/docs/agents/index.html"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/agents/index.html.svg?down_color=red&down_message=offline&up_message=online"></a>
    <a href="https://github.com/huggingface/agents/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/agents.svg"></a>
    <a href="https://github.com/huggingface/agents/blob/main/CODE_OF_CONDUCT.md"><img alt="Contributor Covenant" src="https://img.shields.io/badge/Contributor%20Covenant-v2.0%20adopted-ff69b4.svg"></a>
</p>

<h3 align="center">
<p>Smolagents - build great agents!</p>
</h3>

Smolagents is a library that enables you to run powerful agents in a few lines of code!

This library offers:

✨ **Simplicity**: the logic for agents fits in ~thousand lines of code. We kept abstractions to their minimal shape above raw code!

🌐 **Support for any LLM**: it supports models hosted on the Hub loaded in their `transformers` version or through our inference API, but also models from OpenAI, Anthropic, and many more through our LiteLLM integration.

🧑‍💻 **First-class support for Code Agents**, i.e. agents that write their actions in code (as opposed to "agents being used to write code"), [read more here](tutorials/secure_code_execution).

🤗 **Hub integrations**: you can share and load tools to/from the Hub, and more is to come!

## Quick demo

First install the package.
```bash
pip install agents
```
Then define your agent, give it the tools it needs and run it!
```py
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())

agent.run("What time would the world's fastest car take to travel from New York to San Francisco?")
```

> TODO: Add video

## Code agents?

We built agents where the LLM engine writes its actions in code. This approach is demonstrated to work better than the current industry practice of letting the LLM output a dictionary of the tools it wants to calls: [uses 30% fewer steps](https://huggingface.co/papers/2402.01030) (thus 30% fewer LLM calls)
and [reaches higher performance on difficult benchmarks](https://huggingface.co/papers/2411.01747). Head to [./conceptual_guides/intro_agents.md] to learn more on that.

Especially, since code execution can be a security concern (arbitrary code execution!), we provide options at runtime:
  - a secure python interpreter to run code more safely in your environment
  - a sandboxed environment.

## How lightweight is it?

We strived to keep abstractions to a strict minimum, with the main code in `agents.py` being roughly 1,000 lines of code, and still being quite complete, with several types of agents implemented: `CodeAgent` writing its actions in code snippets, and the more classic `ToolCallingAgent` that leverage built-in tool calling methods.

Many people ask: why use a framework at all? Well, because a big part of this stuff is non-trivial. For instance, the code agent has to keep a consistent format for code throughout its system prompt, its parser, the execution. So our framework handles this complexity for you. But of course we still encourage you to hack into the source code and use only the bits that you need, to the exclusion of everything else!
