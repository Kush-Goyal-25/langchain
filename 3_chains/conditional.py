import os

from dotenv import load_dotenv
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnableSequence,
)
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI

# Load environment variables (e.g., GOOGLE_API_KEY)
load_dotenv()

llm1 = ChatMistralAI(
    model="codestral-2405",  # Paid llm, check available llms at docs.mistral.ai
    temperature=0.7,
    max_tokens=500,
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # Free tier llm
    temperature=0.7,  # Controls randomness (0.0 to 1.0)
    max_tokens=500,  # Limits response length
)

positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a thank you note for this positive feedback: {feedback}."),
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a response addressing this negative feedback: {feedback}."),
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Generate a request for more details for this neutral feedback: {feedback}.",
        ),
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Generate a message to escalate this feedback to a human agent: {feedback}.",
        ),
    ]
)

# Define the feedback classification template
classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}.",
        ),
    ]
)

# Define the runnable branches for handling feedback
branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback_template | llm | StrOutputParser(),  # Positive feedback chain
    ),
    (
        lambda x: "negative" in x,
        negative_feedback_template | llm | StrOutputParser(),  # Negative feedback chain
    ),
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | llm | StrOutputParser(),  # Neutral feedback chain
    ),
    escalate_feedback_template | llm | StrOutputParser(),
)

# Create the classification chain
classification_chain = classification_template | llm | StrOutputParser()

# Combine classification and response generation into one chain
chain = classification_chain | branches

# Run the chain with an example review
# Good review - "The product is excellent. I really enjoyed using it and found it very helpful."
# Bad review - "The product is terrible. It broke after just one use and the quality is very poor."
# Neutral review - "The product is okay. It works as expected but nothing exceptional."
# Default - "I'm not sure about the product yet. Can you tell me more about its features and benefits?"

review = (
    "The product is terrible. It broke after just one use and the quality is very poor."
)
result = chain.invoke({"feedback": review})

# Output the result
print(result)
