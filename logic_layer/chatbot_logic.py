import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI

# Load environment variables from the .env file
load_dotenv()

# Retrieve the OpenAI API key from environment
openai_key = os.getenv("OPENAI_API_KEY")

# Load the vectorstore using the OpenAI embeddings
vectorstore = FAISS.load_local(
    "vectorstore",
    OpenAIEmbeddings(openai_api_key=openai_key),
    allow_dangerous_deserialization=True
)



# Define the chatbot response function
def ask_bot(question: str) -> str:
    # Search for similar documents
    docs = vectorstore.similarity_search(question)

    # Build the context string from the search results
    context = "\n".join([doc.page_content for doc in docs])

    # Format the prompt
    prompt = f"Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion:\n{question}"

    # Initialize the OpenAI LLM
    llm = OpenAI(openai_api_key=openai_key)

    # Get the response from the model
    return llm.invoke(prompt)
