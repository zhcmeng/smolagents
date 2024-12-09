from agents.llm_engine import TransformersEngine
from agents import CodeAgent, ReactJsonAgent

import requests
from datetime import datetime

model_repo="andito/SmolLM2-1.7B-Instruct-F16-GGUF"
model_filename="smollm2-1.7b-8k-dpo-f16.gguf"

import random
from llama_cpp import Llama

model = Llama.from_pretrained(
    repo_id=model_repo,
    filename=model_filename,
    n_ctx=8192,
    verbose=False
)
print("Model initialized")

def llm_engine(messages, stop_sequences=["Task", "<|endoftext|>"]) -> str:
    output = ""
    for chunk in model.create_chat_completion(
        messages=messages,
        max_tokens=2048,
        temperature=0.0,
        top_p=1.0,
        top_k=50,
        repeat_penalty=1.0,
        stream=True
    ):
        content = chunk['choices'][0]['delta'].get('content')
        if content:
            if content in ["<end_action>", "<|endoftext|>"]:
                break
            output += content
    return output

system_prompt = """You are an expert in composing functions. You are given a question and a set of possible functions. 
Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
If none of the functions can be used, point it out and refuse to answer. 
If the given question lacks the parameters required by the function, also point it out.

You have access to the following tools:
<<tool_descriptions>>

<<managed_agents_descriptions>>

You can use imports in your code, but only from the following list of modules: <<authorized_imports>>

The output MUST strictly adhere to the following format, and NO other text MUST be included.
The example format is as follows. Please make sure the parameter type is correct. If no function call is needed, please make the tool calls an empty list '[]'.
<tool_call>[
{"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}},
... (more tool calls as required)
]</tool_call>"""


from agents import tool
import webbrowser

@tool
def get_random_number_between(min: int, max: int) -> int:
    """
    Gets a random number between min and max.

    Args:
        min: The minimum number.
        max: The maximum number.

    Returns:
        A random number between min and max.
    """
    return random.randint(min, max)


@tool
def get_weather(city: str) -> str:
    """
    Returns the weather forecast for a given city.

    Args:
        city: The name of the city.

    Returns:
        A string with a mock weather forecast.
    """
    url = 'https://wttr.in/{}?format=+%C,+%t'.format(city)
    res = requests.get(url).text

    return f"The weather in {city} is {res.split(',')[0]} with a high of {res.split(',')[1][:-2]} degrees Celsius."

@tool
def get_current_time() -> str:
    """
    This is a tool that returns the current time.
    It returns the current time as HH:MM.
    """
    return f"The current time is {datetime.now().hour}:{datetime.now().minute}."

@tool
def open_webbrowser(url: str) -> str:
    """
    This is a tool that opens a web browser to the given website.
    If the user asks to open a website or a browser, you should use this tool.

    Args:
        url: The url to open.
    """
    webbrowser.open(url)
    return f"I opened {url.replace('https://', '').replace('www.', '')} in the browser."


agent = ReactJsonAgent(llm_engine = llm_engine, tools=[get_current_time, open_webbrowser, get_random_number_between, get_weather])
print("Agent initialized!")
agent.run("What's the weather like in London?")