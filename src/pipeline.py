from ingestion import Ingestion
from chunking import Chunking
from indexing import VectorStore
from retrieval import Retrievel
from reranking import CrossEncoderReranker


class AdvanceRAGPipeline:
    def __init__(self, query):
        self.query = query

    def build_db(self):
        ingestion = Ingestion(directory="../data")
        docs = ingestion.load_docs()
        
        chunks = Chunking(documents=docs)
        chunk = chunks.DocumentAwarenessChunking()

        store = VectorStore(chunks=chunk)
        store.create_vector_store()

        return chunk
    
    def run_pipeline(self):
        chunk = self.build_db()

        store = VectorStore(chunks=chunk)
        vector_store = store.load_vector_db()

        retrieval = Retrievel(chunks=chunk, vector_store = vector_store)
        context = retrieval.create_hybrid_retriever()
        candidate = context.invoke(self.query)

        ranking = CrossEncoderReranker()
        reranking_context = ranking.rerank(self.query, candidate, top_n=3)

        return reranking_context