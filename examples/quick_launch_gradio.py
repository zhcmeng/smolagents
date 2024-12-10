from agents.gradio_ui import GradioUI
from agents import HfApiEngine, load_tool, CodeAgent

image_generation_tool = load_tool("m-ric/text-to-image")

llm_engine = HfApiEngine("Qwen/Qwen2.5-72B-Instruct")

agent = CodeAgent(tools=[image_generation_tool], llm_engine=llm_engine)

GradioUI(agent).run()