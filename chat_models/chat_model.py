import os

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables (e.g., GOOGLE_API_KEY)
load_dotenv()

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # Free tier model
    temperature=0.7,  # Controls randomness (0.0 to 1.0)
    max_tokens=500,  # Limits response length
)

# Use .invoke() with a list of HumanMessage objects
result = llm.invoke([HumanMessage(content="What is the capital of France?")])
print(result.content)  # Access the response content
