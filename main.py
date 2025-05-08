import gradio as gr
from main1 import get_inference_response  # assuming this returns (response, response_time)

def chat_interface(user_input, history):
    if not user_input:
        return gr.update(), history, ""  # Don't process empty input

    response, _ = get_inference_response(user_input)

    # # Remove any source document info if present
    # if "Source Document" in response:
    #     response = response.split("Source Document")[0].strip()

    history.append((user_input, response))
    return history, history, ""  # Update chatbot, state, clear textbox

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="Chatbot")
    state = gr.State([])  # Maintain chat history properly

    with gr.Row():
        with gr.Column(scale=8):
            txt = gr.Textbox(show_label=False, placeholder="Ask something about Data Science").style(container=False)
        with gr.Column(scale=1):
            submit_btn = gr.Button("Send")

    clear_btn = gr.Button("Clear Chat")

    submit_btn.click(
        chat_interface,
        inputs=[txt, state],
        outputs=[chatbot, state, txt]
    )

    txt.submit(
        chat_interface,
        inputs=[txt, state],
        outputs=[chatbot, state, txt]
    )

    clear_btn.click(
        lambda: ([], [], ""),
        outputs=[chatbot, state, txt]
    )

demo.launch(server_port=8080, server_name="localhost")
