import requests
import os
import base64
from app.core.bootstrap import bootstrap_runtime

bootstrap_runtime()

api_key = os.getenv("GOOGLE_API_KEY")

payload = {
    "contents": [
        {
            "parts": [
                {"text": "What is 1+1?"}
            ]
        }
    ]
}

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
r = requests.post(url, json=payload)
print(r.status_code, r.text)
