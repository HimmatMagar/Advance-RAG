# Advanced RAG

An implementation-focused project exploring **Advanced Retrieval-Augmented Generation (RAG)** techniques, including different chunking strategies, retrieval methods, hybrid search, and reranking to improve the relevance and quality of retrieved context.

## Overview

Traditional RAG systems often struggle with irrelevant search results, poor document segmentation, and retrieving information that is semantically similar but not actually useful for answering a query.

This project explores different strategies to address these challenges and understand how each component affects retrieval performance.

## Features

### 1. Document Ingestion
- Load documents from PDF, TXT, and Markdown files.
- Extract document content and preserve source metadata.
- Prepare documents for downstream processing.

### 2. Chunking Strategies
Experiment with different approaches to split documents into meaningful chunks.

- **Fixed-size chunking:** Split text into chunks of a specified size.
- **Document-aware chunking:** Use document structure and separators to preserve meaningful context.
- **Semantic chunking:** Split text based on semantic relationships between content.

### 3. Retrieval Techniques
Explore different methods of retrieving relevant information from a vector database.

- **Top-k retrieval:** Retrieve a fixed number of the most relevant chunks.
- **Similarity-based retrieval:** Rank chunks based on their semantic similarity to a query.
- **Similarity threshold:** Filter out results that do not meet a specified relevance threshold.
- **Metadata filtering:** Restrict retrieval using document metadata.

### 4. Hybrid Search
Combine two complementary retrieval approaches:

- **BM25:** Keyword-based retrieval that matches important terms in the query.
- **Semantic search:** Vector-based retrieval that identifies meaning and contextual similarity.
- **Ensemble retrieval:** Combine results from multiple retrievers using weighted reciprocal rank fusion.

Hybrid search helps retrieve documents that keyword search or semantic search alone might miss.

### 5. Reranking
Use a cross-encoder reranker to refine the initial retrieval results.

Unlike bi-encoder embedding models, which independently encode queries and documents, a cross-encoder evaluates the query and each candidate document together to produce a more precise relevance score.

This allows the system to reorder retrieved chunks so that the most relevant context appears first.

## Architecture

The general retrieval pipeline follows this workflow:

```text
Documents (PDF / TXT / MD)
          |
          v
    Document Ingestion
          |
          v
    Chunking Strategies
          |
          v
      Embeddings
          |
          v
      Vector Store
          |
User Query --> Hybrid Retrieval
                     |
                     v
                Reranking
                     |
                     v
              Relevant Context
                     |
                     v
           LLM Answer Generation
```

*Note: The diagram represents the overall Advanced RAG workflow. The exact stages depend on the pipeline configuration and the components enabled in the implementation.*

## Technology Stack

- **Python** — Core implementation
- **LangChain** — Document processing and retrieval components
- **ChromaDB** — Vector storage and similarity search
- **Sentence Transformers** — Embedding and reranking models
- **BM25** — Keyword-based retrieval
- **PyPDF** — PDF document processing

## Project Structure

The project is organized around separate components for document processing, indexing, retrieval, and reranking.

```text
Advance-RAG/
├── data/                 # Source documents
├── src/
│   ├── ingestion.py      # Document loading
│   ├── chunking.py       # Chunking strategies
│   ├── indexing.py       # Vector store and indexing
│   ├── retrieval.py      # Retrieval and hybrid search
│   ├── reranking.py      # Cross-encoder reranking
│   └── pipeline.py       # Pipeline orchestration
├── CHROMA_DB/            # Persistent vector database
├── pyproject.toml        # Project dependencies and configuration
├── uv.lock               # Locked dependencies
└── README.md
```

*Adjust the file paths above to match the actual repository structure.*

## Installation

### Prerequisites

- Python
- [uv](https://docs.astral.sh/uv/)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/HimmatMagar/Advance-RAG.git
cd Advance-RAG
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Add documents

Place your supported PDF, TXT, or Markdown documents in the project's data directory.

### 4. Run the pipeline

Run the appropriate entry point for your implementation. For example:

```bash
uv run python src/pipeline.py
```

Update this command if your pipeline uses a different entry point or requires a query argument.

## Learning Objectives

This project is designed to explore:

- How chunk size, overlap, and document structure affect retrieval.
- The trade-offs between keyword-based and semantic retrieval.
- How hybrid search combines complementary retrieval signals.
- Why initial retrieval results may contain irrelevant documents.
- How cross-encoder reranking improves candidate ordering.
- How to design modular, testable RAG pipelines.

## Future Improvements

- Implement retrieval and generation evaluation using metrics such as Recall@k, MRR, and nDCG.
- Compare chunking strategies using a consistent evaluation dataset.
- Tune hybrid retrieval weights and reranking candidate counts.
- Add contextual compression and parent-document retrieval.
- Evaluate answer faithfulness, relevance, and groundedness.
- Add tracing, latency measurement, and experiment tracking.
- Integrate an LLM for end-to-end question answering.

## Author

**Himmat Magar**

GitHub: [@HimmatMagar](https://github.com/HimmatMagar)

---

*This project is part of my hands-on exploration of Advanced RAG architectures and practical information retrieval techniques.*