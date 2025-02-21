import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# 🔹 โหลด BERT Model สำหรับสร้าง Embeddings
model = SentenceTransformer("sentence-transformers/bert-base-nli-mean-tokens")

# 🔹 โหลดข้อมูล Web Scraping
df = pd.read_csv("scraped_data.csv")
texts = df["content"].tolist()

# 🔹 แปลงเป็น Embeddings
embeddings = model.encode(texts)

# 🔹 สร้าง FAISS Index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings, dtype=np.float32))

# 🔹 บันทึก Index
faiss.write_index(index, "faiss_index.bin")
print("✅ FAISS Index สร้างสำเร็จ!")
