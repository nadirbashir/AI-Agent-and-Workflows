import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

def get_temperature(city: str) -> float:
    print(f"[TOOL] Fetching temperature for: {city}")
    return 20.0

available_functions = {
    "get_temperature": get_temperature,
}

get_temperature_function = {
    "name": "get_temperature",
    "description": "Get current temperature for a given location.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City for which to get the temperature."
            }
        },
        "required": ["city"],
    },
}

tools = types.Tool(function_declarations=[get_temperature_function])

config = types.GenerateContentConfig(
    tools=[tools]
)

def execute_tool_call(name, args):
    if name in available_functions:
        try:
            return available_functions[name](**args)
        except Exception as e:
            return f"Error: {e}"
    return "Unknown tool"


def run_agent(chat_history):
    while True:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=chat_history,
            config=config,
        )

        candidate = response.candidates[0]
        parts = candidate.content.parts

        chat_history.append(
            types.Content(
                role="model",
                parts=parts
            )
        )

        function_call = None

        for part in parts:
            if part.function_call:
                function_call = part.function_call

        if not function_call:
            for part in parts:
                if part.text:
                    print(part.text)
            return

        fn_name = function_call.name
        fn_args = function_call.args

        print(f"Agent Calling tool: {fn_name} with {fn_args}")

        result = execute_tool_call(fn_name, fn_args)

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

def main():
    chat_history = []

    while True:
        user_input = input("\nYour question (type 'exit'): ")

        if user_input.lower() == "exit":
            break

        chat_history.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_input)]
            )
        )

        run_agent(chat_history)

        print("-" * 50)


if __name__ == "__main__":
    main()