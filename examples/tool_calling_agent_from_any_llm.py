from smolagents.agents import ToolCallingAgent
from smolagents import tool, HfApiModel, TransformersModel, LiteLLMModel

# Choose which LLM engine to use!
model = HfApiModel("meta-llama/Llama-3.3-70B-Instruct")
model = TransformersModel("meta-llama/Llama-3.2-2B-Instruct")
model = LiteLLMModel("gpt-4o")

@tool
def get_weather(location: str) -> str:
    """
    Get weather in the next days at given location.
    Secretly this tool does not care about the location, it hates the weather everywhere.

    Args:
        location: the location
    """
    return "The weather is UNGODLY with torrential rains and temperatures below -10°C"

agent = ToolCallingAgent(tools=[get_weather], model=model)

print(agent.run("What's the weather like in Paris?"))