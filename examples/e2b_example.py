from agents import Tool, CodeAgent
from agents.default_tools.search import VisitWebpageTool
from dotenv import load_dotenv
load_dotenv()

LAUNCH_GRADIO = False

class GetCatImageTool(Tool):
    name="get_cat_image"
    description = "Get a cat image"
    inputs = {}
    output_type = "image"

    def __init__(self):
        super().__init__()
        self.url = "https://em-content.zobj.net/source/twitter/53/robot-face_1f916.png"

    def forward(self):
        from PIL import Image
        import requests
        from io import BytesIO

        response = requests.get(self.url)

        return Image.open(BytesIO(response.content))

get_cat_image = GetCatImageTool()


agent = CodeAgent(
    tools = [get_cat_image, VisitWebpageTool()],
    additional_authorized_imports=["Pillow", "requests", "markdownify"], # "duckduckgo-search", 
    use_e2b_executor=False
)

if LAUNCH_GRADIO:
    from agents.gradio_ui import GradioUI

    GradioUI(agent).launch()
else:
    agent.run(
        "Return me an image of Lincoln's preferred pet",
        additional_context="Here is a webpage about US presidents and pets: https://www.9lives.com/blog/a-history-of-cats-in-the-white-house/"
    )
