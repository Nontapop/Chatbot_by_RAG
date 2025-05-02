from flask import Flask, request, jsonify, render_template
from retriever import retrieve_documents
from huggingface_hub import InferenceClient
import os

# 🔹 ใช้ Access Token และโมเดลจาก Hugging Face
API_TOKEN = os.getenv("HF_TOKEN")  # 🔸 ใช้ Token ของคุณ
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

client = InferenceClient(model=MODEL_NAME, token=API_TOKEN)

app = Flask(__name__)

# 🔹 หน้าเว็บหลัก
@app.route("/")
def home():
    return render_template("index.html")

# 🔹 API สำหรับถามคำถาม
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    query = data.get("query")
    if not query:
        return jsonify({"error": "Query is required!"}), 400
    
    # 🔹 ดึงเอกสารที่เกี่ยวข้องจาก RAG (FAISS + Retriever)
    relevant_docs = retrieve_documents(query)
    context = " ".join(relevant_docs)  # รวมข้อความที่ได้มาเป็นบริบท (Context)

    # 🔹 ใช้โมเดล Mistral ในการสร้างคำตอบจากบริบทที่ดึงมา
    prompt = f"คำถาม: {query}\nเนื้อหาที่เกี่ยวข้อง: {context}\nคำตอบ:"
    response = client.text_generation(prompt, max_new_tokens=200)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

