#!/usr/bin/env python
# coding: utf-8

# In[6]:


import warnings
warnings.filterwarnings('ignore')


# In[7]:


from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader


# In[4]:


def load_docs(directory: str):
    documents = []

    for path in Path(directory).rglob('*'):
        if not path.is_file():
            continue

        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(file_path=str(path))
            docs = loader.load()

        elif path.suffix.lower() in {".txt", ".md"}:
            loader = TextLoader(file_path=str(path), encoding='UTF-8')
            docs = loader.load()

        else:
            continue

        for doc in docs:
            doc.metadata['source'] = str(path)
            doc.metadata['file_name'] = path.name

        documents.extend(docs)
    return documents


# In[5]:


loaded_documents = load_docs(directory="../data")
loaded_documents


# In[11]:


sources = {d.metadata['source'] for d in loaded_documents}
sources


# In[13]:


print(f"Files loaded: {len(sources)}")
for s in sources:
    print(" -", s)
print(f"Total Document objects: {len(loaded_documents)}")


# In[ ]:




