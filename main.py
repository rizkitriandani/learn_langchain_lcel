from langchain.prompts import PromptTemplate
from getpass import getpass
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

question = "Make a poem about Jakarta in max payne style."

for chunk in llm.stream("Tell me a short poem about snow"):
    sys.stdout.write(chunk)
    sys.stdout.flush()
    
# print(chain.invoke({"question": question}))
