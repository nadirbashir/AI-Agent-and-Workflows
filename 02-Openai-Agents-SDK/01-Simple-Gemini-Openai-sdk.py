import os

from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled

load_dotenv()

set_tracing_disabled(disabled=True)
external_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

llm_model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

agent = Agent(name="Assistant", model=llm_model)

result = Runner.run_sync(starting_agent=agent, input="Welcome and motivate me to learn Agentic AI briefly in 5-6 sentences.")

print("AGENT RESPONSE: " , result.final_output)

