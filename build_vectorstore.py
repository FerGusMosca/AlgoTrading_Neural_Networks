import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# Set path to the documents folder
docs_folder = "input/docs"
if not os.path.exists(docs_folder):
    raise FileNotFoundError(f"The folder '{docs_folder}' does not exist.")

# Load all .docx files in the folder
documents = []
for filename in os.listdir(docs_folder):
    if filename.endswith(".docx"):
        filepath = os.path.join(docs_folder, filename)
        print(f"📄 Loading {filename}...")
        loader = UnstructuredWordDocumentLoader(filepath)
        documents.extend(loader.load())

if not documents:
    raise ValueError("No .docx documents were loaded. Check the path and file formats.")

# Split text into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Print sample chunk for debug
print(f"🔍 Preview of first chunk:\n{texts[0].page_content[:300]}...\n")

# Build vectorstore with OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
db = FAISS.from_documents(texts, embeddings)

# Save vectorstore locally
db.save_local("vectorstore")
print("✅ Vectorstore built and saved successfully.")
