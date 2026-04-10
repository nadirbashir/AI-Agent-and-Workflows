import os
import sqlite3
from datetime import datetime, timedelta
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
    """
    Verifies a customer's identity using their name and PIN.
    Returns the customer ID if verified, or -1 if not found.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    first_name, last_name = name.lower().split()
    cursor.execute(
        "SELECT id FROM customers WHERE LOWER(first_name) = ? AND LOWER(last_name) = ? AND pin = ?",
        (first_name, last_name, pin),
    )
    result = cursor.fetchone()
    conn.close()
    if result:
        return result[0]
    return -1


def get_orders(customer_id: int) -> List[dict]:
    """
    Retrieves the order history for a given customer.
    """
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM orders WHERE customer_id = ?", (customer_id,))
    orders = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return orders


def check_refund_eligibility(customer_id: int, order_id: int) -> bool:
    """
    Checks if an order is eligible for a refund.
    An order is eligible if it was placed within the last 30 days.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT date FROM orders WHERE id = ? AND customer_id = ?", (
            order_id, customer_id)
    )
    result = cursor.fetchone()
    conn.close()
    if not result:
        return False
    order_date = datetime.fromisoformat(result[0])
    return (datetime.now() - order_date).days <= 30


def issue_refund(customer_id: int, order_id: int) -> bool:
    """
    Issues a refund for an order.
    """
    # in reality, this would be stored in some database
    print(f"Refund issued for order {order_id} for customer {customer_id}")
    return True


def share_feedback(customer_id: int, feedback: str) -> str:
    """
    Allows a customer to share feedback.
    """
    # in reality, this would be stored in some database
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
                "name": {
                    "type": "string",
                    "description": "The customer's full name, e.g., 'John Doe'.",
                },
                "pin": {"type": "string", "description": "The customer's PIN."},
            },
            "required": ["name", "pin"],
        },
    },
    {
        "name": "get_orders",
        "description": "Retrieves the order history for a verified customer.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "integer",
                    "description": "The customer's unique ID.",
                }
            },
            "required": ["customer_id"],
        },
    },
    {
        "name": "check_refund_eligibility",
        "description": "Checks if an order is eligible for a refund based on the order date.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "integer",
                    "description": "The customer's unique ID.",
                },
                "order_id": {
                    "type": "integer",
                    "description": "The unique ID of the order.",
                },
            },
            "required": ["customer_id", "order_id"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Issues a refund for an order.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "integer",
                    "description": "The customer's unique ID.",
                },
                "order_id": {
                    "type": "integer",
                    "description": "The unique ID of the order.",
                },
            },
            "required": ["customer_id", "order_id"],
        },
    },
    {
        "name": "share_feedback",
        "description": "Allows a customer to provide feedback about their experience.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {
                    "type": "integer",
                    "description": "The customer's unique ID.",
                },
                "feedback": {
                    "type": "string",
                    "description": "The feedback text from the customer.",
                },
            },
            "required": ["customer_id", "feedback"],
        },
    },
]


tools = types.Tool(function_declarations=tool_funcs)

config = types.GenerateContentConfig(
    tools=[tools]
)


def execute_tool_call(name, args) -> str:
    """
    Executes a tool call and returns the output.
    """

    if name in available_functions:
        function_to_call = available_functions[name]
        try:
            print(f"Calling {name} with arguments: {args}")
            # The return value of the function is converted to a string to be compatible with the API.
            return available_functions[name](**args)
        except Exception as e:
            return f"Error calling {name}: {e}"

    return f"Unknown tool: {name}"


def main():
    instructions = """
                You are a friendly and helpful customer service agent. 
                You must ALWAYS verify the customer's identity before providing any sensitive information. 
                You MUST NOT expose any information to unverified customers.
                You MUST NOT provide any information that is not related to the customer's question.
                DON'T guess any information - neither customer nor order related (or anything else).
                If you can't perform a certain customer or order-related task, you must direct the user to a human agent.
                Ask for confirmation before performing any key actions.
                If you can't help a customer or if a customer is asking for something that is not related to the customer service, you MUST say "I'm sorry, I can't help with that."
            """
    messages = [
        types.Content(
                role="model",
                parts=[types.Part.from_text(text=instructions)]
            )
    ]

    print("Welcome to the customer service chatbot! How can we help you today? Please type 'exit' to end the conversation.")
    while True:
        user_input = input(
            "Your input: ")
        if user_input == "exit":
            break

        messages.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_input)]
            )
        )

        for _ in range(5):  # limit tool call / assistant cycles to prevent infinite loops
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
                contents=messages,
                config=config,
            )

            candidate = response.candidates[0]
            parts = candidate.content.parts

            function_call = None

            for part in parts:
                if part.function_call:
                    function_call = part.function_call

            if not function_call:
                for part in parts:
                    if part.text:
                        print(part.text)

            fn_name = function_call.name
            fn_args = function_call.args

            result = execute_tool_call(fn_name, fn_args)
            messages.append(
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

            if not isinstance(messages[-1], dict) and messages[-1].type == "message":
                break
if __name__ == "__main__":
    main()