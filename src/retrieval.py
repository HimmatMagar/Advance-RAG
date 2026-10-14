from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever



class Retrievel:
    def __init__(self, chunks: list[Document], vector_store):
        self.chunks = chunks
        self.vector_store = vector_store
    
    
    def create_hybrid_retriever(self) -> EnsembleRetriever:
        bm25_retriver = BM25Retriever.from_documents(self.chunks)
        bm25_retriver.k = 5

        vector_retriver = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )

        hybrid_retriver = EnsembleRetriever(
            retrievers=[bm25_retriver, vector_retriver],
            weights=[0.5, 0.5],
            c = 60
        )
        return hybrid_retriver