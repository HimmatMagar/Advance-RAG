#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


get_ipython().run_line_magic('run', 'ingestion.ipynb')
document = load_docs(directory="../data")


# In[3]:


document[-1]


# In[4]:


get_ipython().run_line_magic('run', 'chunking.ipynb')
chunks = DocumentAwarenessChunking(document)


# In[5]:


chunks[1].metadata


# In[6]:


for chunk in chunks[:3]:
    print("\nSOURCE:", chunk.metadata.get("source"))
    print("CONTENT:", repr(chunk.page_content[:500]))


# In[7]:


get_ipython().run_line_magic('run', 'indexing.ipynb')
vector_store = create_vector_store(chunks)


# In[10]:


query = "What is Recurrent Neural Networks"


# In[11]:


get_ipython().run_line_magic('run', 'retrieval.ipynb')
hybrid_retriever = create_hybrid_retriever(chunks, vector_store)


# In[12]:


candidates = hybrid_retriever.invoke(query)


# In[13]:


print("Candidates:", len(candidates))


# In[16]:


get_ipython().run_line_magic('run', 'reranking.ipynb')
reranker = CrossEncoderReranker()


# In[17]:


results = reranker.rerank(
    query=query,
    documents=candidates,
    top_n=5,
)


# In[18]:


for rank, item in enumerate(results, start=1):
    doc = item["document"]

    print(f"\nRank: {rank}")
    print(f"Reranker score: {item['score']:.4f}")
    print(f"Source: {doc.metadata.get('source')}")
    print(f"Page: {doc.metadata.get('page')}")
    print(doc.page_content[:700])


# In[ ]:




