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
# Introduction to Agents

[[open-in-colab]]

### Why do we need agentic systems?

Current LLMs are like basic reasoning robots, that are trapped into a room.
They can be sometimes impressively smart – and often impressively dumb – but they can only take as input what we decide to provide to them. We pass notes under the door – be it text, or text with images for vision models, or even audio –, and they reply to each note by passing another note under the door, but they cannot do anything else.

Wouldn't it be much more efficient to let them have some kind of access to the real world, either as a way to do their own research in order to better answer a question, or a way to accomplish a complex task for us?

In other words, give them some agency.

The whole idea of agentic systems is to embed LLMs into a program where their input and outputs are optimized to better leverage real-world interactions.


### What is an agentic system ?

Being "agentic" is not a 0-1 definition: instead, we should talk about "agency", defined as a spectrum.

Any system leveraging LLMs will embed them into code. Then the influence of the LLM's input on the code workflow is the level of agency of LLMs in the system.

If the output of the LLM has no further impact on the way functions are run, this system is not agentic at all.

Once one an LLM output is used to determine which branch of an `if/else` switch is ran, that starts to be some level of agency: a router.

Then it can get more agentic.

If you use an LLM output to determine which function is run and with which arguments, that's tool calling.

If you use an LLM output to determine if you should keep iterating in a while loop, you get a multi-step agent.

And the workflow can become even more complex. That's up to you to decide.



### Why {Agents}?

For some low-level agentic use cases, like chains or routers, you can write all the code yourself. You'll be much better that way, since it will let you control and understand your system better.

But once you start going for more complicated behaviours like letting an LLM call a function (that's "tool calling") or letting an LLM run a while loop ("multi-step agent"), some abstractions become necessary:
- for tool calling, you need to parse the agent's output, so this output needs a predefined format like "Thought: I should call tool 'get_weather'. Action: get_weather(Paris).", that you parse with a predefined function, and system prompt given to the LLM should notify it about this format.
- for a multi-step agent where the LLM output determines the loop, you need to give a different prompt to the LLM based on what happened in the last loop iteration: so you need some kind of memory.

See? With these two examples, we already found the need for a few items to help us:
- a list of tools that the agent can access
- a parser that extracts tool calls from the LLM output
- system prompt synced with the parser
- memory

### Most important feature: Code agent

[Multiple](https://huggingface.co/papers/2402.01030) [research](https://huggingface.co/papers/2411.01747) papers have shown that having the LLM write its actions (the tool calls) in code is much better than the current standard format JSON.

Why is that? Well, because we crafted our code languages specifically to be great at expressing actions performed by a computer. If JSON snippets was a better way, this package would have been written in JSON snippets and the devil would be having a great time laughing at us.

Code is just a better way to express actions on a computer. It has better:
- Composability: could you nest JSON actions within each other, or define a set of JSON actions to re-use later, the same way you could just define a python function?
- Object management: how do you store the output of an action like `generate_image` in JSON?
- Generality: code is made to express simply anything you can do have a computer do.

So we decided to give you the best Code agents out there!