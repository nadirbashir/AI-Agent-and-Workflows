import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import gradio as gr
import requests

load_dotenv()

class GeminiAgent:

    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        self.available_functions = {
            "get_temperature": self.get_temperature,
        }

        self.tools = types.Tool(
            function_declarations=[
                {
                    "name": "get_temperature",
                    "description": "Get current temperature for a given location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "lat": {
                                "type": "string",
                                "description": "latitude of the location."
                            },
                            "lon": {
                                "type": "string",
                                "description": "longitude of the location."
                            }
                        },
                        "required": ["lat", "lon"],
                    },
                }
            ]
        )

        self.config = types.GenerateContentConfig(
            tools=[self.tools]
        )

    def system_prompt(self):
        return """
You are a smart AI assistant.

Your goals:
- Be concise, clear, and helpful
- Use tools ONLY when necessary
- Never hallucinate tool results
- If a tool is available and relevant, prefer using it over guessing
- After using a tool, explain the result naturally in a friendly way to the user

Tool usage rules:
- Always pass correct arguments with the location latitude and longitude when calling get_temperature
- Do not call tools unnecessarily
- If the user asks for real-world data (weather, temperature, wind, rainfall, humidity etc.), use tools

Response style:
- Keep answers short unless user asks for details
- Be conversational but not verbose
"""

 
    def get_temperature(self, lat: str, lon: str) -> float:
        print(f"[TOOL] Fetching temperature for: {lat}, {lon}")
        response = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m"
        )
        data = response.json()
        return data["current"]
    
   

    def execute_tool_call(self, name, args):
        if name in self.available_functions:
            try:
                return self.available_functions[name](**args)
            except Exception as e:
                return f"Error: {e}"
        return "Unknown tool"

    def run_agent(self, chat_history):
        while True:
            response = self.client.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=chat_history,
                config=self.config,
            )

            candidate = response.candidates[0]
            parts = candidate.content.parts

            chat_history.append(
                types.Content(role="model", parts=parts)
            )

            function_call = None

            for part in parts:
                if part.function_call:
                    function_call = part.function_call

            if not function_call:
                final_text = ""
                for part in parts:
                    if part.text:
                        final_text += part.text
                return final_text

            fn_name = function_call.name
            fn_args = function_call.args

            print(f"[AGENT] Calling tool: {fn_name} with {fn_args}")

            result = self.execute_tool_call(fn_name, fn_args)

            chat_history.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name=fn_name,
                            response={"result": result}
                        )
                    ]
                )
            )

    def chat(self, message, history):
        chat_history = []

        chat_history.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=self.system_prompt())]
            )
        )

        for h in history:
            chat_history.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=h["content"])]
                )
            )

        chat_history.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=message)]
            )
        )

        response = self.run_agent(chat_history)

        return response


if __name__ == "__main__":
    agent = GeminiAgent()

    gr.ChatInterface(
        fn=agent.chat,
        type="messages",
        title="AI Agent with real-time weather data",
        description="Ask anything. The agent can use tools when needed."
    ).launch()