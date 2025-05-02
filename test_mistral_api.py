from dotenv import load_dotenv
import os
import requests

# โหลด environment variables จากไฟล์ .env
load_dotenv()

# ตั้งค่าตัวแปร API URL ของ Hugging Face API
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"

# ดึงค่า API Token จาก environment variable
API_TOKEN = os.getenv("HF_TOKEN")

# ตรวจสอบว่า Token ถูกต้องหรือไม่
if not API_TOKEN:
    raise ValueError("API Token is missing! Please set the HF_TOKEN environment variable.")

# กำหนด header สำหรับการร้องขอ API
headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

# ข้อมูลที่ส่งไปยัง API
payload = {
    "inputs": "What is the capital of Thailand?"
}

# ส่งคำขอไปยัง API
response = requests.post(API_URL, headers=headers, json=payload)

# ตรวจสอบสถานะการตอบกลับจาก API
print("Status Code:", response.status_code)

# ถ้า status code ไม่ใช่ 200 ให้ดูข้อความที่ได้จาก response
if response.status_code != 200:
    print("Error: Received status code", response.status_code)
    print("Response Text:", response.text)  # ดูข้อความจาก response
else:
    try:
        # แสดงผลการตอบกลับในรูปแบบ JSON
        response_json = response.json()
        print("Response:", response_json)
    except requests.exceptions.JSONDecodeError as e:
        print("Error in decoding JSON:", e)
