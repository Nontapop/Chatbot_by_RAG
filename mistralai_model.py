from huggingface_hub import InferenceClient

# ใช้ Access Token ของคุณ
API_TOKEN = ""  # 🔹 เปลี่ยนเป็น Token ของคุณ
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# สร้าง Client
client = InferenceClient(model=MODEL_NAME, token=API_TOKEN)

# Prompt ที่ต้องการให้โมเดลตอบ
prompt = "เอกสารที่ใช้ในการสมัครเรียน"

# เรียกใช้งาน API
response = client.text_generation(prompt, max_new_tokens=200)

# แสดงผลลัพธ์
print(response)
