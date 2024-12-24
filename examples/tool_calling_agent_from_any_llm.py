from smolagents.agents import ToolCallingAgent
from smolagents import tool, HfApiEngine, OpenAIEngine, AnthropicEngine, TransformersEngine, LiteLLMEngine

# Choose which LLM engine to use!
# llm_engine = OpenAIEngine("gpt-4o")
# llm_engine = AnthropicEngine("claude-3-5-sonnet-20240620")
# llm_engine = HfApiEngine("meta-llama/Llama-3.3-70B-Instruct")
# llm_engine = TransformersEngine("meta-llama/Llama-3.2-2B-Instruct")
llm_engine = LiteLLMEngine("gpt-4o")

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