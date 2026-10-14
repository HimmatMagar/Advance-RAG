from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunking:
    def __init__(self, documents: list[Document]):
        self.documents = documents

    def DocumentAwarenessChunking(self) -> list[Document]:
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

        chunks = splitter.split_documents(self.documents)
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = index

        return chunks


    def ParentChildChunking(self) -> tuple[list[Document], list[Document]]:
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

        parent_chunks = parent_splitter.split_documents(self.document)
        all_child_chunks = []
        child_id = 0

        for parent_idx, parent in enumerate(parent_chunks):
            parent.metadata['parent_id'] = parent_idx

            children = child_splitter.split_documents([parent])

            for child in children:
                child.metadata["parent_id"] = parent_idx
                child.metadata["chunk_id"] = child_id
                child_id += 1
                all_child_chunks.append(child)

        return parent_chunks, all_child_chunks