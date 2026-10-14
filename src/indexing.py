from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

PRESIST_DIR = "CHROMA_DB"
COLLECTION_NAME = 'Advance-RAG'
class VectorStore:

    def __init__(self, chunks):
        self.embeddings = HuggingFaceEmbeddings(
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.chunks = chunks

    def create_vector_store(self) -> Chroma:
        """Creating new DB from documents"""
        vector_store = Chroma.from_documents(
            documents=self.chunks,
            embedding=self.embeddings,
            persist_directory=PRESIST_DIR,
            collection_name=COLLECTION_NAME
        )

        return vector_store

    
    def load_vector_db(self) -> Chroma:
        """Loading Existing DB from local disk"""
        vector_store = Chroma(
            persist_directory=PRESIST_DIR,
            embedding_function=self.embeddings,
            collection_name=COLLECTION_NAME
        )
        return vector_store