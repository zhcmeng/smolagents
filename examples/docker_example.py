from agents.search import DuckDuckGoSearchTool
from agents.docker_alternative import DockerPythonInterpreter

container = DockerPythonInterpreter()

tools = [DuckDuckGoSearchTool]

output = container.execute("res = web_search(query='whats the capital of Cambodia?'); print(res)", tools=tools)

print(output)

container.stop()
