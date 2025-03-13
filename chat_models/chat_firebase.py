import os

from dotenv import load_dotenv
from google.cloud import firestore
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI

# Load environment variables (e.g., GOOGLE_API_KEY)
load_dotenv()

PROJECT_ID = "langchain-1662f"
SESSION_ID = "user_session_new"
COLLECTION_NAME = "chat_history"

print("Initialize Firestore client...")
client = firestore.Client(project=PROJECT_ID)

print("Initialize Firestore chat message history...")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,
)
print("Chat history initialized.")
print("Current chat history:", chat_history.messages)

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


print("Start chatting with the AI assistant. Type 'exit' to end the conversation.")
while True:
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break

    chat_history.add_user_message(human_input)

    result = llm.invoke(chat_history.messages)
    chat_history.add_ai_message(result.content)

    print(f"AI: {result.content}")
