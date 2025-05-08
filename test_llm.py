from llama_cpp import Llama

llm = Llama(
    model_path="C:/Darshil/SDE/LLM/llm_gguf_model/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_gpu_layers=20,  # Set this depending on your GPU VRAM (adjust higher/lower)
    n_ctx=2048        # Context size
)

response = llm("Q: What is the capital of France?\nA:")
print(response)