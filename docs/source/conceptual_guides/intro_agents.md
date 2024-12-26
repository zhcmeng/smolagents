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

### What is an agent?

Current LLMs are like basic reasoning robots that are trapped into a room.
They take as input what we decide to provide to them. We pass notes under the door – be it text, or text with images for vision models, or even audio –, and they reply to each note by passing another note under the door, but they cannot do anything else.

Any efficient system using AI will need to provide LLMs some kind of access to the real world: for instance the possibility to call a search tool to get external information, or to act on certain programs in order to solve a task.

In other words, give them some agency. Agentic programs are the gateway to the outside world for LLMs.

Our definition of AI Agents is : “programs in which the workflow is determined by LLM outputs”. Any system leveraging LLMs will embed them into code. The influence of the LLM's input on the code workflow is the level of agency of LLMs in the system.

Note that with this definition, "agent" is not a discrete, 0 or 1 definition: instead, "agency" evolves on a continuous spectrum, as you give more or less influence to the LLM on your workflow.


If the output of the LLM has no impact on the workflow, as in a program that just postprocesses a LLM's output and returns it, this system is not agentic at all.

If an LLM output is used to determine which branch of an `if/else` switch is ran, the system starts to have some level of agency: it's a router.

Then it can get more agentic.
- If you use an LLM output to determine which function is run and with which arguments, that's tool calling.
- If you use an LLM output to determine if you should keep iterating in a while loop, you get a multi-step agent.

| Agency Level | Description | How that's called | Example Pattern |
|-------------|-------------|-------------|-----------------|
| No Agency | LLM output has no impact on program flow | Simple Processor | `process_llm_output(llm_response)` |
| Basic Agency | LLM output determines basic control flow | Router | `if llm_decision(): path_a() else: path_b()` |
| Higher Agency | LLM output determines function execution | Tool Caller | `run_function(llm_chosen_tool, llm_chosen_args)` |
| High Agency | LLM output controls iteration and program continuation | Multi-step Agent | `while llm_should_continue(): execute_next_step()` |
| High Agency | One agentic workflow can start another agentic workflow | Multi-Agent | `if llm_trigger(): execute_agent()` |

Since the system’s versatility goes in lockstep with the level of agency that you give to the LLM, agentic systems can perform much broader tasks than any classic program.

Programs are not just tools anymore, confined to an ultra-specialized task : they are agents.


### When to use an agentic system ?

Agents are useful when you need an LLM to help you determine the workflow of an app.

It's advise to regularize towards not using any agentic behaviour.
Ask yourself: do I really need flexibility in the workflow to efficiently solve the task at hand? If a fixed workflow would work, you might as well build it all in good old no-AI code for 100% robustness. Agents are useful when the ficed workflow is not enough.

Let's take an example: say you're making an app that handles customer requests on a surfing trip website.

You could know in advance that the requests will have to be classified in either of 2 buckets according to deterministic criteria, and you have a predefined workflow for each of these 2 cases.
For instance, this is if you let the user click a button to determine their query, and it goes into either of these buckets:
1. Want some knowledge on the trips. Then you give them access to a search bar to search your knowledge base
2. Wants to talk to sales. Then you let them type in a contact form.

If that deterministic workflow fits all queries, by all means just code verything: this will give you a 100% reliable system with no risk of error introduced by letting unpredictable LLMs meddle in your workflow.

But what if the workflow can't be determined that well in advance? Say, 10% or 20% of users requests do not fit properly into your rigid categories, and are thus not handled properly by your program?

Let's say, a user wants to ask : "I can come on Monday, but I forgot my passport so risk being delayed to Wednesday, is it possible to take me and my stuff to surf on Tuesday morning, with a concellation insurance?"
This question into play many factors: availability of employees, weather, travelling distance, knowledge about cancellation policies...
Probably none of the predetermined criteria above won't work properly on this question.

If these cases where the predetermined workflow falls short are frequent, that means you need more flexibility: making your system agentic will provide it that flexibility. In our example, you could just make a multi-step agent that has access to a weather API tool, a google maps API to compute travel distance, an employee availability dashboard and a RAG system on your knowledge base.

Actually, most real-life tasks do not fit in a pre-determined workflow. This is why until today, our programs where always focused on infinitely narrow tasks, like "compute the sum of these numbers" or "find the shortest path in this graph". 

Agentic systems are a great way to introduce the vast world of real-world tasks to programs!

### Why Smolagents?

For some low-level agentic use cases, like chains or routers, you can write all the code yourself. You'll be much better that way, since it will let you control and understand your system better.

But once you start going for more complicated behaviours like letting an LLM call a function (that's "tool calling") or letting an LLM run a while loop ("multi-step agent"), some abstractions become necessary:
- for tool calling, you need to parse the agent's output, so this output needs a predefined format like "Thought: I should call tool 'get_weather'. Action: get_weather(Paris).", that you parse with a predefined function, and system prompt given to the LLM should notify it about this format.
- for a multi-step agent where the LLM output determines the loop, you need to give a different prompt to the LLM based on what happened in the last loop iteration: so you need some kind of memory.

See? With these two examples, we already found the need for a few items to help us:
- of course an LLM that acts as the engine powering the system
- a list of tools that the agent can access
- a parser that extracts tool calls from the LLM output
- system prompt synced with the parser
- memory
But wait, since we give room to LLMs in decisions, surely they will make mistakes, so for better performance we need error logging and retry mechanism?

These will not be that straightforward to implement correctly, especially not together. That's why we decided that we needed to build a few abstractions to help people use these.

### Code agents

[Multiple](https://huggingface.co/papers/2402.01030) [research](https://huggingface.co/papers/2411.01747) [papers](https://huggingface.co/papers/2401.00812) have shown that having the LLM write its actions (the tool calls) in code is much better than the current standard format for tool calling, which is across the industry different shades of "writing actions as a JSON of tools names and arguments to use".

Why is code better? Well, because we crafted our code languages specifically to be great at expressing actions performed by a computer. If JSON snippets was a better way, this package would have been written in JSON snippets and the devil would be laughing at us.

Code is just a better way to express actions on a computer. It has better:
- **Composability:** could you nest JSON actions within each other, or define a set of JSON actions to re-use later, the same way you could just define a python function?
- **Object management:** how do you store the output of an action like `generate_image` in JSON?
- **Generality:** code is built to express simply anything you can do have a computer do.
- **Representation in LLM training corpuses:** why not leverage this benediction of the sky that plenty of quality actions have already been included in LLM training corpuses?

This is illustrated on the figure below, taken from [Executable Code Actions Elicit Better LLM Agents](https://huggingface.co/papers/2402.01030).

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/code_vs_json_actions.png">

This is why we put emphasis on proposing code agents, in this case python agents, which meant building secure python interpreters.