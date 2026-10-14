#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


# In[3]:


def create_hybrid_retriever(chunks, vector_store):
    bm25_retriver = BM25Retriever.from_documents(chunks)
    bm25_retriver.k = 5

    vector_retriver = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    hybrid_retriver = EnsembleRetriever(
        retrievers=[bm25_retriver, vector_retriver],
        weights=[0.5, 0.5],
        c = 60
    )
    return hybrid_retriver


# In[4]:


get_ipython().run_line_magic('run', 'ingestion.ipynb')
document = load_docs(directory="../data")


# In[5]:


get_ipython().run_line_magic('run', 'chunking.ipynb')
chunks = DocumentAwarenessChunking(document)


# In[7]:


get_ipython().run_line_magic('run', 'indexing.ipynb')
vector_store = create_vector_store(chunks)


# In[8]:


hybrid_retriever = create_hybrid_retriever(chunks, vector_store)


# In[9]:


query = "What is Recurrent Neural Networks"


# In[11]:


results = hybrid_retriever.invoke(query)

for rank, doc in enumerate(results, start=1):
    print(f"\nRank: {rank}")
    print(f"Source: {doc.metadata.get('source')}")
    print(f"Page: {doc.metadata.get('page')}")
    print(doc.page_content[:1000])
    print("-" * 60)


# In[ ]:




