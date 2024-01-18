from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os, sys


# Load the .env file
load_dotenv()

# Get the API key
api_key = os.getenv("API_KEY")

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate.from_template(template)

llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

chain = prompt | llm

while True:
    question = input("Please enter your question or 'exit' to quit: ")

    if question.lower() == "exit":
        break

    for chunk in llm.stream(question):
        sys.stdout.write(chunk)
        sys.stdout.flush()


# print(chain.invoke({"question": question}))
