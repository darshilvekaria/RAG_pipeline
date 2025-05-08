from sentence_transformers import SentenceTransformer
import os
import certifi
import shutil


os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['SSL_CERT_FILE'] = certifi.where()

model_name = "hkunlp/instructor-base"
save_path = "./models_instructor/instructor-base"

# Create the directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

# Delete existing model folder if it exists. it is done to remove corrupt files as it wont be overwritten during model creation
if os.path.exists(save_path):
    shutil.rmtree(save_path)


# Download the model using sentence-transformers (includes tokenizer + pooling logic)
print(f"Downloading and saving model: {model_name}")
model = SentenceTransformer(model_name)


# Save everything (model + tokenizer + config)
model.save(save_path)
print(f"Model saved successfully to {save_path}")
