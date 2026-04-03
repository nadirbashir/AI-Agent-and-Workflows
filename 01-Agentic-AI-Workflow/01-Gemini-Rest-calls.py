import os
import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def generate_post(topic: str) -> str:
    prompt = {
        "contents": [
        {
            "parts": [
            {
                "text": topic
            }
            ]
        }
        ]
    }
    response = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent",
    headers = {
        "content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
        },
    json=prompt
    )
    reply = response.json()
    print(reply)
    return reply["candidates"][0]["content"]["parts"][0]["text"]

def main():
    usr_input = input("ask something?")
    x_post = generate_post(usr_input)
    print(x_post)

if __name__ == "__main__":
    main()