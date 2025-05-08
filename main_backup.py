import gradio as gr
from main1 import get_inference_response  # assuming this returns (response, response_time)

def chat_interface(user_input, history=[]):
    response, _ = get_inference_response(user_input)
    history.append((user_input, response))
    return history, history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=8):
            txt = gr.Textbox(show_label=False, placeholder="Ask Data Science Questions").style(container=False)
        with gr.Column(scale=1):
            submit_btn = gr.Button("Send")

    clear_btn = gr.Button("Clear Chat")

    submit_btn.click(chat_interface, inputs=[txt, chatbot], outputs=[chatbot, chatbot])
    txt.submit(chat_interface, inputs=[txt, chatbot], outputs=[chatbot, chatbot])
    clear_btn.click(lambda: [], None, chatbot)

demo.launch(server_port=8080, server_name="localhost")
