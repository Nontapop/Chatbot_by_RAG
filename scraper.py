import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_website(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    
    # ดึง text content
    paragraphs = soup.find_all("p")
    text_content = " ".join([p.text for p in paragraphs])

    return text_content

# 🔹 ดึงข้อมูลจากหลายเว็บ
urls = [
    "https://www.bu.ac.th/th/faq/bachelor",
]
data = [scrape_website(url) for url in urls if scrape_website(url) is not None]

# 🔹 บันทึกเป็น CSV
df = pd.DataFrame({"content": data})
df.to_csv("scraped_data.csv", index=False)

print("✅ Web Scraping เสร็จสิ้น บันทึกข้อมูลเรียบร้อย!")

