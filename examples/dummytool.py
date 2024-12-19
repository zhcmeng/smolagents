from agents.tool import Tool


class DummyTool(Tool):
    name = "echo"
    description = """Perform a web search based on your query (think a Google search) then returns the top search results as a list of dict elements.
    Each result has keys 'title', 'href' and 'body'."""
    inputs = {
        "cmd": {"type": "string", "description": "The search query to perform."}
    }
    output_type = "any"

    def forward(self, cmd: str) -> str:
       return cmd 