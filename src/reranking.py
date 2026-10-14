#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import torch
from sentence_transformers import CrossEncoder


# In[9]:


class CrossEncoderReranker:
    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="mps",
    ):
        self.model = CrossEncoder(
            model_name,
            device=device,
        )

    def rerank(self, query, documents, top_n=5):
        if not documents:
            return []

        pairs = [
            (query, doc.page_content)
            for doc in documents
        ]

        scores = self.model.predict(
            pairs,
            batch_size=8,
            show_progress_bar=False,
        )

        ranked = sorted(
            zip(scores, documents),
            key=lambda item: float(item[0]),
            reverse=True,
        )

        return [
            {
                "score": float(score),
                "document": doc,
            }
            for score, doc in ranked[:top_n]
        ]


# In[ ]:




