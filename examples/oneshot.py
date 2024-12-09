from agents import load_tool, CodeAgent, JsonAgent, HfApiEngine
from agents.prompts import ONESHOT_CODE_SYSTEM_PROMPT

# Import tool from Hub
image_generation_tool = load_tool("m-ric/text-to-image", cache=False)

# Import tool from LangChain
from agents.search import DuckDuckGoSearchTool

search_tool = DuckDuckGoSearchTool()

llm_engine = HfApiEngine("Qwen/Qwen2.5-Coder-32B-Instruct")
# Initialize the agent with both tools
agent = CodeAgent(
    tools=[image_generation_tool, search_tool],
    llm_engine=llm_engine,
    system_prompt=ONESHOT_CODE_SYSTEM_PROMPT,
    verbose=True
)

# Run it!
result = agent.run(
    "When was Llama 3 first released?"
)

print(result)