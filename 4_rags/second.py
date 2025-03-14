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

# Configure Google GenAI SDK using environment variable

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # Free tier model
    temperature=0.7,  # Controls randomness (0.0 to 1.0)
    max_tokens=500,  # Limits response length
)

# Paths for file and persistent storage
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Initialize embeddings with a specific model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Check if vector store exists, initialize if not
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Example file path (adjust as needed)
    file_path = os.path.join(current_dir, "documents", "example.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please provide a valid text file."
        )

    # Load and split the document
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    # Create and persist the vector store
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print("Vector store initialized.")
else:
    # Load the existing vector store
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
    print("Loaded existing vector store.")

# Define the user's question
query = "Where does Gandalf meet Frodo?"

# Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 10, "score_threshold": 0.7},  # Lowered threshold
)
relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
if not relevant_docs:
    print("No documents found above the similarity threshold.")
else:
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
