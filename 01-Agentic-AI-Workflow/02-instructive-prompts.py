import os
import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def generate_post(topic: str) -> str:
    prompt = {
        "system_instruction": {
            "parts": [
                {
                "text": "You are an expert social media manager, and you excel at crafting viral and high engaging posts for X (formerly Twitter)."
                },
                {
                "text": "Your task is to generate post that is concise, impactful, and tailored to the topic provided by the user."
                },
                {
                "text": "Avoid using hashtags and lots of emojis (a few emojis are okay, but not too many)."
                },
                {
                "text": "Keep the post brief and focused, structure it in a clean, readable way, using line breaks and empty lines to enhance readability."
                }
            ]
        },
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
    
    try:
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContnt",
            headers={
                "Content-Type": "application/json",
                "X-goog-api-key": GEMINI_API_KEY
            },
            json=prompt,
        )

        reply = response.json()
        return reply["candidates"][0]["content"]["parts"][0]["text"]

    except Exception as e:
        print("Error:", e)
        return None

def main():
    usr_input = input("what should be the post about?")
    x_post = generate_post(usr_input)
    print(x_post)
if __name__ == "__main__":
    main()