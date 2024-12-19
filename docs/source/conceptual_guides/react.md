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
# ReAct agents

## One shot agent

This agent has a planning step, then generates python code to execute all its actions at once. It natively handles different input and output types for its tools, thus it is the recommended choice for multimodal tasks.

## React agents

This is the go-to agent to solve reasoning tasks, since the ReAct framework ([Yao et al., 2022](https://huggingface.co/papers/2210.03629)) makes it really efficient to think on the basis of its previous observations.

We implement two versions of JsonAgent: 
- [`JsonAgent`] generates tool calls as a JSON in its output.
- [`CodeAgent`] is a new type of JsonAgent that generates its tool calls as blobs of code, which works really well for LLMs that have strong coding performance.

> [!TIP]
> Read [Open-source LLMs as LangChain Agents](https://huggingface.co/blog/open-source-llms-as-agents) blog post to learn more about ReAct agents.

<div class="flex justify-center">
    <img
        class="block dark:hidden"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Agent_ManimCE.gif"
    />
    <img
        class="hidden dark:block"
        src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/Agent_ManimCE.gif"
    />
</div>

![Framework of a React Agent](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-source-llms-as-agents/ReAct.png)