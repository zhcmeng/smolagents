from agents import stream_to_gradio, HfApiEngine, load_tool, CodeAgent
import gradio as gr

image_generation_tool = load_tool("m-ric/text-to-image")

llm_engine = HfApiEngine("Qwen/Qwen2.5-72B-Instruct")

agent = CodeAgent(tools=[image_generation_tool], llm_engine=llm_engine)

def interact_with_agent(prompt, messages):
    messages.append(gr.ChatMessage(role="user", content=prompt))
    yield messages
    for msg in stream_to_gradio(agent, task=prompt, reset_agent_memory=False):
        messages.append(msg)
        yield messages
    yield messages


with gr.Blocks() as demo:
    stored_message = gr.State([])
    chatbot = gr.Chatbot(label="Agent",
                         type="messages",
                         avatar_images=(None, "https://em-content.zobj.net/source/twitter/53/robot-face_1f916.png"))
    text_input = gr.Textbox(lines=1, label="Chat Message")
    text_input.submit(lambda s: (s, ""), [text_input], [stored_message, text_input]).then(interact_with_agent, [stored_message, chatbot], [chatbot])

demo.launch()