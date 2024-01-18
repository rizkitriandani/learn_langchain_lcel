from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from typing import Optional
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import (
    RunnableWithMessageHistory,
    RunnableLambda,
    RunnablePassthrough,
)
from operator import itemgetter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
import os, sys
import uuid


def main():
    # Load the .env file
    load_dotenv()

    # Get the API key
    api_key = os.getenv("API_KEY")
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=api_key
    )

    loader = PyPDFLoader("data/HD - Generator - Akhmad Rizki Triandani.pdf")
    pages = loader.load()

    # vectorstore = FAISS.from_documents(pages, embedding=embedding)

    # vectorstore.save_local("data/vectorstore")

    vectorstore = FAISS.load_local("data/vectorstore", embeddings=embedding)

    llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

    retriever = vectorstore.as_retriever()

    template = """Answer the question based only on the following context:
    {context}
    
    and give recommendation to take action based on it, in giving the recommendation you can take real world examples.

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    for chunk in chain.stream(
        "apa tipe human design saya? dan apa saja karakteristiknya?"
    ):
        sys.stdout.write(chunk)
        sys.stdout.flush()
    sys.stdout.write("\n")


# docs = vectorstore.similarity_search(
#     "apa cara pengambilan keputusan berdasarkan human design?", k=2
# )

# print(f"docs: {docs}")

# for doc in docs:
#     print(str(doc.metadata["page"]) + ":", doc.page_content[:300])

# retriever = vectorstore.as_retriever()


if __name__ == "__main__":
    main()
