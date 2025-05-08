from transformers import AutoTokenizer, AutoModel
import os
import certifi
import shutil


os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['SSL_CERT_FILE'] = certifi.where()

model_name = "hkunlp/instructor-base"
save_path = "./models_instructor/instructor-base"


# Delete existing model folder if it exists. it is done to remove corrupt files as it wont be overwritten during model creation
if os.path.exists(save_path):
    shutil.rmtree(save_path)

# Create the directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

# Download tokenizer and model
print(f"Downloading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, cache_dir="./tmp_cache")

# Save them locally
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
print(f"Model saved to {save_path}")
