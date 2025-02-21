import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd

# 🔹 โหลด FAISS Index และข้อมูล
index = faiss.read_index("faiss_index.bin")
df = pd.read_csv("scraped_data.csv")
texts = df["content"].tolist()
model = SentenceTransformer("sentence-transformers/bert-base-nli-mean-tokens")

def retrieve_documents(query, top_k=3):
    query_embedding = model.encode([query])
    _, indices = index.search(np.array(query_embedding, dtype=np.float32), top_k)
    
    relevant_docs = [texts[i] for i in indices[0]]
    return relevant_docs
