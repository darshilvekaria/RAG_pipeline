
import box
import yaml
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from InstructorEmbedding import INSTRUCTOR
import os
import certifi


os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['SSL_CERT_FILE'] = certifi.where()


# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


# Build vector database
def run_db_build():
    loader = DirectoryLoader(cfg.DATA_PATH,
                            glob='*.pdf',
                            show_progress=True,
                            recursive=True,
                            loader_cls=PyPDFLoader)
    documents = loader.load()
    print('document loaded successfully')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.CHUNK_SIZE,
                                                   chunk_overlap=cfg.CHUNK_OVERLAP)
    texts = text_splitter.split_documents(documents)
    print('text split completed')

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cuda'})

    # Create the FAISS vector store
    vectorstore = FAISS.from_documents(texts, embeddings)

    vectorstore.save_local(cfg.DB_FAISS_PATH)
    print('Vector database created and saved')

if __name__ == "__main__":
    run_db_build()
