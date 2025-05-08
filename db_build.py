from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from src.InstructorEmbeddingWrapper import InstructorEmbeddingWrapper

import os
import yaml
import box
import certifi

os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['SSL_CERT_FILE'] = certifi.where()

# Load config
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

def run_db_build():
    loader = DirectoryLoader(
        cfg.DATA_PATH,
        glob="*.pdf",
        show_progress=True,
        recursive=True,
        loader_cls=PyPDFLoader,
    )
    documents = loader.load()
    print("Documents loaded successfully.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(documents)
    print("Text splitting completed.")

    #  Use the working wrapper
    model_path = "models_instructor/instructor-base"
    print(f"Loading InstructorEmbedding model from {model_path}...")
    embeddings = InstructorEmbeddingWrapper(model_path, device="cuda")

    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(cfg.DB_FAISS_PATH)
    print(f"Vector store saved to {cfg.DB_FAISS_PATH}.")


if __name__ == "__main__":
    run_db_build()
