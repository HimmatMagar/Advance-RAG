#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# In[ ]:


def create_vector_store(chunks):
    embedding = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory="chroma_db",
        collection_name="advance-rag"
    )

    return vector_store


# In[ ]:




