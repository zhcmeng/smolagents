from agents.default_tools.search import DuckDuckGoSearchTool
from agents.docker_alternative import DockerPythonInterpreter


from agents.tools import Tool

class DummyTool(Tool):
    name = "echo"
    description = '''Perform a web search based on your query (think a Google search) then returns the top search results as a list of dict elements.
    Each result has keys 'title', 'href' and 'body'.'''
    inputs = {
        "cmd": {"type": "string", "description": "The search query to perform."}
    }
    output_type = "any"

    def forward(self, cmd: str) -> str:
       return cmd 


container = DockerPythonInterpreter()

output = container.execute("x = 5")
print(f"first output: {output}")
output = container.execute("print(x)")
print(f"second output: {output}")

breakpoint()

print("---------")


output = container.execute("res = DummyTool(cmd='echo this'); print(res())")
print(output)

container.stop()
