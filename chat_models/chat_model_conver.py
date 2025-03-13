import os

from dotenv import load_dotenv
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI

# Load environment variables (e.g., GOOGLE_API_KEY)
load_dotenv()

llm1 = ChatMistralAI(
    model="codestral-2405",  # Paid model, check available models at docs.mistral.ai
    temperature=0.7,
    max_tokens=500,
)


# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # Free tier model
    temperature=0.7,  # Controls randomness (0.0 to 1.0)
    max_tokens=500,  # Limits response length
)

chat_history = []

system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message)


messages = [
    HumanMessage(content="Give a short tip to create a engaging post on instagram"),
    SystemMessage(content="You are expert in social media marketing"),
]

result = llm.invoke(messages)

print(result.content)  # Access the response content
