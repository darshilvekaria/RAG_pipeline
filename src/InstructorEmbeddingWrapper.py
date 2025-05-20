from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

# Custom embedding class using SentenceTransformer for Instructor
class InstructorEmbeddingWrapper(Embeddings):
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model = SentenceTransformer(model_path, device=device)

    def embed_documents(self, texts):
        # Apply Instructor-style instructions
        pairs = [("Represent the document for retrieval:", text) for text in texts]
        return self.model.encode(pairs, normalize_embeddings=True)

    def embed_query(self, text):
        return self.model.encode([("Represent the question for retrieving supporting documents:", text)],
                                 normalize_embeddings=True)[0]
