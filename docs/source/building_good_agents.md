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
# Building good agents

There's a world of difference between building an agent that works and one that doesn't. How to build into this latter category?
In this guide, we're going to see best practices for building agents.

> [!TIP]
> If you're new to `agents`, make sure to first read the [intro to agents](./intro_agents).

### The best agentic systems are the simplest: simplify the workflow as much as you can

Giving an LLM some agency in your workflow introducessome risk of errors.

Well-programmed agentic systems have good error logging and retry mechanisms anyway, so the LLM engine has a chance to self-correct their mistake. But to reduce the risk of LLM error to the maximum, you should simplify your worklow!

Let's take again the example from [intro_agents]: a bot that answers user queries on a surf trip company.
Instead of letting the agent do 2 different calls for "travel distance API" and "weather API" each time they are asked about a new surf spot, you could just make one unified tool "return_spot_information", a functions that calls both APIs at once and returns their concatenated outputs to the user.

This will reduce costs, latency, and error risk!

The main guideline is: Reduce the number of LLM calls as much as you can.

This leads to a few takeaways:
- Whenever possible, group 2 tools in one, like in our example of the two APIs.
- Whenever possible, logic should be based on deterministic functions rather than agentic decisions.

### Improve the information flow to the LLM engine

Remember that your LLM engine is like a ~intelligent~ robot, tapped into a room with the only communication with the outside world being notes passed under a door.

It won't know of anything that happened if you don't explicitly put that into its prompt.

Particular guidelines to follow:
- Each tool should log (by simply using `print` statements inside the tool's `forward` method) everything that could be useful for the LLM engine.
  - In particular, logging detail on tool execution errors would help a lot!

For instance, here's a tool that :

First, here's a poor version:
```py
from my_weather_api return convert_location_to_coordinates, get_weather_report_at_coordinates
# Let's say "get_weather_report_at_coordinates" returns a list of [temperature in °C, risk of rain on a scale 0-1, wave height in m]
import datetime

@tool
def get_weather_api(location (str), date_time: str) -> str:
    """
    Returns the weather report.

    Args:
        - location (`str`): the name of the place that you want the weather for.
        - date_time (`str`): the date and time for which you want the report.
    """
    lon, lat = convert_location_to_coordinates(location)
    date_time = datetime.strptime(date_time)
    return str(get_weather_report_at_coordinates((lon, lat), date_time))
```

Why is it bad?
- there's no precision of the format that should be used for `date_time`
- there's no detail on how location should
- there's no logging mechanism tying to explicit failure cases like location not being in a proper format, or date_time not being properly formatted.
- the output format is hard to understand

If the tool call fails, the error trace logged in memory can help the LLM reverse engineer the tool to fix the errors. But why leave it so much heavy lifting to do?

A better way to build this tool would have been the following:
```py
from my_weather_api return convert_location_to_coordinates, get_weather_report_at_coordinates
# Let's say "get_weather_report_at_coordinates" returns a list of [temperature in °C, risk of rain on a scale 0-1, wave height in m]
import datetime

@tool
def get_weather_api(location (str), date_time: str) -> str:
    """
    Returns the weather report.

    Args:
        - location (`str`): the name of the place that you want the weather for. Should be a place name, followed by possibly a city name, then a country, like "Anchor Point, Taghazout, Morocco".
        - date_time (`str`): the date and time for which you want the report, formatted as '%m/%d/%y %H:%M:%S'.
    """
    lon, lat = convert_location_to_coordinates(location)
    try:
        date_time = datetime.strptime(date_time)
    except Exception as e:
        raise ValueError("Conversion of `date_time` to datetime format failed, make sure to provide a string in format '%m/%d/%y %H:%M:%S'. Full trace:" + str(e))
    temperature_celsius, risk_of_rain, wave_height = get_weather_report_at_coordinates((lon, lat), date_time)
    return f"Weather report for {location}, {date_time}: Temperature will be {temperature_celsius}°C, risk of rain is {risk_of_rain*100:.0f}%, wave height is {wave_height}m."
```

In general, to ease the load on your LLM, the good question to ask yourself is: "How easy would it be for me, if I was dumb and using this tool for thsefirst time ever, to program with this tool and correct my own errors?".

### How to debug your agent

In an agentic workflows, some of the errors are actual errors, some other are the fault of your LLM engine not reasoning properly.
For instance, consider this trace for an `CodeAgent` that I asked to make me a car picture:
```
==================================================================================================== New task ====================================================================================================
Make me a cool car picture
──────────────────────────────────────────────────────────────────────────────────────────────────── New step ────────────────────────────────────────────────────────────────────────────────────────────────────
Agent is executing the code below: ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
image_generator(prompt="A cool, futuristic sports car with LED headlights, aerodynamic design, and vibrant color, high-res, photorealistic")
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

Last output from code snippet: ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
/var/folders/6m/9b1tts6d5w960j80wbw9tx3m0000gn/T/tmpx09qfsdd/652f0007-3ee9-44e2-94ac-90dae6bb89a4.png
Step 1:

- Time taken: 16.35 seconds
- Input tokens: 1,383
- Output tokens: 77
──────────────────────────────────────────────────────────────────────────────────────────────────── New step ────────────────────────────────────────────────────────────────────────────────────────────────────
Agent is executing the code below: ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
final_answer("/var/folders/6m/9b1tts6d5w960j80wbw9tx3m0000gn/T/tmpx09qfsdd/652f0007-3ee9-44e2-94ac-90dae6bb89a4.png")
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Print outputs:

Last output from code snippet: ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
/var/folders/6m/9b1tts6d5w960j80wbw9tx3m0000gn/T/tmpx09qfsdd/652f0007-3ee9-44e2-94ac-90dae6bb89a4.png
Final answer:
/var/folders/6m/9b1tts6d5w960j80wbw9tx3m0000gn/T/tmpx09qfsdd/652f0007-3ee9-44e2-94ac-90dae6bb89a4.png
```
The user sees, instead of an image being returned, a path being returned to them.
It could look like a bug from the system, but actually the agentic system didn't cause the error: it's just that the LLM engine tid the mistake of not saving the image output into a variable.
Thus it cannot access the image again except by leveraging the path that was logged while saving the image, so it returns the path instead of an image.

The first step to debugging your agent is thus "Use a more powerful LLM". Alternatives like `Qwen2/5-72B-Instruct` wouldn't have made that mistake.

Then you can also use less powerful models but guide them better.

To provide extra information, we do not recommend modifying the system prompt compared to default : there are many adjustments there that you do not want to mess up except if you understand the prompt very well.
Better ways to guide your LLM engine are:
- If it 's about the task to solve: add all these details to the task. The task could be 100s of pages long.
- If it's about how to use tools: the description attribute of your tools.


