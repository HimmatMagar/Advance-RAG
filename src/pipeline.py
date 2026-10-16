from pathlib import Path
from ingestion import Ingestion
from chunking import Chunking
from indexing import VectorStore
from retrieval import Retrievel
from reranking import CrossEncoderReranker
from generation import Generation


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

class AdvanceRAGPipeline:
    def __init__(self, query):
        self.query = query

    def build_db(self):
        ingestion = Ingestion(directory=DATA_DIR)
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

        generate = Generation(query= self.query, reranked_results=reranking_context)
        answer = generate.generate_answer()
        return answer.answer


if __name__ == "__main__":
    query = input("Give your question: ")
    rag = AdvanceRAGPipeline(query=query)
    print(rag.run_pipeline())