import warnings
warnings.filterwarnings("ignore")

from langchain_text_splitters import RecursiveCharacterTextSplitter


def DocumentAwarenessChunking(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        add_start_index=True,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks


def ParentChildChunking(documents):
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    parent_chunks = parent_splitter.split_documents(documents)
    child_chunks = child_splitter.split_documents(documents)
    return parent_chunks, child_chunks