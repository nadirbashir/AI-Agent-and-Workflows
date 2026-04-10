import os
import sqlite3
from datetime import datetime
from typing import List

from dotenv import load_dotenv
from google import genai
from google.genai import types

from database_09 import create_db_and_tables

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

DB_FILE = "dummy_database.db"
create_db_and_tables()


def verify_customer(name: str, pin: str) -> int:
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    parts = name.lower().split()
    if len(parts) < 2:
        conn.close()
        return -1

    first_name, last_name = parts[0], parts[1]

    cursor.execute(
        "SELECT id FROM customers WHERE LOWER(first_name) = ? AND LOWER(last_name) = ? AND pin = ?",
        (first_name, last_name, pin),
    )
    result = cursor.fetchone()
    conn.close()

    return result[0] if result else -1


def get_orders(customer_id: int) -> List[dict]:
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM orders WHERE customer_id = ?", (customer_id,))
    orders = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return orders


def check_refund_eligibility(customer_id: int, order_id: int) -> bool:
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT date FROM orders WHERE id = ? AND customer_id = ?",
        (order_id, customer_id),
    )
    result = cursor.fetchone()
    conn.close()

    if not result:
        return False

    order_date = datetime.fromisoformat(result[0])
    return (datetime.now() - order_date).days <= 30


def issue_refund(customer_id: int, order_id: int) -> bool:
    print(f"Refund issued for order {order_id} for customer {customer_id}")
    return True


def share_feedback(customer_id: int, feedback: str) -> str:
    print(f"Feedback received from customer {customer_id}: {feedback}")
    return "Thank you for your feedback!"


available_functions = {
    "verify_customer": verify_customer,
    "get_orders": get_orders,
    "check_refund_eligibility": check_refund_eligibility,
    "issue_refund": issue_refund,
    "share_feedback": share_feedback,
}

tool_funcs = [
    {
        "name": "verify_customer",
        "description": "Verifies a customer's identity using their full name and PIN.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "pin": {"type": "string"},
            },
            "required": ["name", "pin"],
        },
    },
    {
        "name": "get_orders",
        "description": "Retrieves order history.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "integer"},
            },
            "required": ["customer_id"],
        },
    },
    {
        "name": "check_refund_eligibility",
        "description": "Check if order is refundable.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "integer"},
                "order_id": {"type": "integer"},
            },
            "required": ["customer_id", "order_id"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Issue refund.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "integer"},
                "order_id": {"type": "integer"},
            },
            "required": ["customer_id", "order_id"],
        },
    },
    {
        "name": "share_feedback",
        "description": "Store feedback.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "integer"},
                "feedback": {"type": "string"},
            },
            "required": ["customer_id", "feedback"],
        },
    },
]

tools = types.Tool(function_declarations=tool_funcs)

config = types.GenerateContentConfig(tools=[tools])

def execute_tool_call(name, args):
    if name in available_functions:
        try:
            print(f"[TOOL] Calling {name} with {args}")
            result = available_functions[name](**args)
            return {"result": result} 
        except Exception as e:
            return {"error": str(e)}
    return {"error": f"Unknown tool: {name}"}


def main():
    instructions = """
You are a customer support agent.

Rules:
- ALWAYS verify customer first
- NEVER expose data without verification
- Ask before issuing refunds
- Use tools for real data
- Be concise and helpful
"""

    messages = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=instructions)],
        )
    ]

    print("Welcome! Type 'exit' to quit.")

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            break

        messages.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_input)],
            )
        )

        for _ in range(5):
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=messages,
                config=config,
            )

            candidate = response.candidates[0]
            parts = candidate.content.parts

            messages.append(
                types.Content(
                    role="model",
                    parts=parts,
                )
            )

            function_call = None

            for part in parts:
                if part.function_call:
                    function_call = part.function_call

            if not function_call:
                for part in parts:
                    if part.text:
                        print("\nAgent:", part.text)
                break

            fn_name = function_call.name
            fn_args = function_call.args

            if fn_name not in available_functions:
                print(f"Unknown tool: {fn_name}")
                break

            result = execute_tool_call(fn_name, fn_args)

            messages.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name=fn_name,
                            response=result,
                        )
                    ],
                )
            )


if __name__ == "__main__":
    main()