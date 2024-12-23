from agents.agents import ToolCallingAgent
from agents import tool, HfApiEngine, OpenAIEngine, AnthropicEngine

# Choose which LLM engine to use!
llm_engine = OpenAIEngine("gpt-4o")
llm_engine = AnthropicEngine("claude-3-5-sonnet-20240620")
llm_engine = HfApiEngine("meta-llama/Llama-3.3-70B-Instruct")

@tool
def get_weather(location: str) -> str:
    """
    Get weather in the next days at given location.
    Secretly this tool does not care about the location, it hates the weather everywhere.

    Args:
        location: the location
    """
    return "The weather is UNGODLY with torrential rains and temperatures below -10°C"

agent = ToolCallingAgent(tools=[get_weather], llm_engine=llm_engine)

print(agent.run("What's the weather like in Paris?"))