from agents import OpenAIEngine, AnthropicEngine, HfApiEngine, CodeAgent
from dotenv import load_dotenv

load_dotenv()

openai_engine = OpenAIEngine(model_name="gpt-4o")

agent = CodeAgent([], llm_engine=openai_engine)

print("\n\n##############")
print("Running OpenAI agent:")
agent.run("What is the 10th Fibonacci Number?")


anthropic_engine = AnthropicEngine()

agent = CodeAgent([], llm_engine=anthropic_engine)

print("\n\n##############")
print("Running Anthropic agent:")
agent.run("What is the 10th Fibonacci Number?")

# Here, our token stored as HF_TOKEN environment variable has accesses 'Make calls to the serverless Inference API' and 'Read access to contents of all public gated repos you can access'
llama_engine = HfApiEngine(model="meta-llama/Llama-3.3-70B-Instruct")

agent = CodeAgent([], llm_engine=llama_engine)

print("\n\n##############")
print("Running Llama3.3-70B agent:")
agent.run("What is the 10th Fibonacci Number?")
