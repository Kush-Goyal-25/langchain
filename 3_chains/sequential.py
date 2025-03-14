import os

from dotenv import load_dotenv
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel, RunnableSequence
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI

# Load environment variables (e.g., GOOGLE_API_KEY)
load_dotenv()

llm1 = ChatMistralAI(
    model="codestral-2405",  # Paid model, check available models at docs.mistral.ai
    temperature=0.7,
    max_tokens=500,
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # Free tier model
    temperature=0.7,  # Controls randomness (0.0 to 1.0)
    max_tokens=500,  # Limits response length
)

summary_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie critic."),
        ("human", "Provide a brief summary of the movie {movie_name}."),
    ]
)


def analyze_plot(plot):
    plot_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a movie critic."),
            (
                "human",
                "Analyze the plot: {plot}. What are its strengths and weaknesses?",
            ),
        ]
    )
    return plot_template.format_prompt(plot=plot)


# Define character analysis step
def analyze_characters(characters):
    character_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a movie critic."),
            (
                "human",
                "Analyze the characters: {characters}. What are their strengths and weaknesses?",
            ),
        ]
    )
    return character_template.format_prompt(characters=characters)


# Combine analyses into a final verdict
def combine_verdicts(plot_analysis, character_analysis):
    return (
        f"Plot Analysis:\n{plot_analysis}\n\nCharacter Analysis:\n{character_analysis}"
    )


plot_branch_chain = RunnableLambda(lambda x: analyze_plot(x)) | llm | StrOutputParser()


character_branch_chain = (
    RunnableLambda(lambda x: analyze_characters(x)) | llm | StrOutputParser()
)

chain = (
    summary_template
    | llm
    | StrOutputParser()
    | RunnableParallel(
        branches={"plot": plot_branch_chain, "characters": character_branch_chain}
    )
    | RunnableLambda(
        lambda x: combine_verdicts(x["branches"]["plot"], x["branches"]["characters"])
    )
)

response = chain.invoke({"movie_name": "The Godfather"})

print(response)
