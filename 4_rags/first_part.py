import os

import google.generativeai as genai
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_mistralai import ChatMistralAI

# Load environment variables (e.g., GOOGLE_API_KEY)
load_dotenv()

# Configure Google GenAI SDK
genai.configure(api_key=os.getenv("AIzaSyAzwVso-3ORabshaTFCja2WNcl5YCq8SnU"))

# Initialize the Mistral model (unused here but corrected)
llm1 = ChatMistralAI(
    model="codestral-latest",  # Updated to a likely valid model
    temperature=0.7,
    max_tokens=500,
)

# Initialize the Gemini chat model
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",  # Corrected to free tier model
    temperature=0.7,  # Controls randomness (0.0 to 1.0)
    max_tokens=500,  # Limits response length
)

# Paths for file and persistent storage
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "documents", "lord_of_the_rings.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Check if vector store exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )

    # Read the text content from the file
    loader = TextLoader(file_path)
    documents = loader.load()

    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    # Create embeddings using Google's free embedding model
    print("\n--- Creating embeddings ---")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")

else:
    print("Vector store already exists. No need to initialize.")
