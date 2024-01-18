from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from typing import Optional
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
import os, sys
import uuid


def main():
    # Load the .env file
    load_dotenv()

    # Get the API key
    api_key = os.getenv("API_KEY")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're an assistant who helps people to understand things using simple language, analogy and examples",
            ),
            ("human", "{question}"),
        ]
    )

    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

    chain = prompt | llm

    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: RedisChatMessageHistory(
            session_id, url="redis://localhost:6379/0"
        ),
        input_messages_key="question",
    )

    session_id = str(uuid.uuid4())

    while True:
        question = input("Please enter your question or 'exit' to quit: ")

        if question.lower() == "exit":
            break

        for chunk in chain_with_history.stream(
            {
                "question": question,
            },
            config={"configurable": {"session_id": session_id}},
        ):
            sys.stdout.write(chunk)
            sys.stdout.flush()
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
