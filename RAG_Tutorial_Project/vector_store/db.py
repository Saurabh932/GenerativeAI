from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

PDF_PATH = "document_loaders/Git Notes.pdf"
DB_PATH = "chroma-db"
COLLECTION_NAME = "git_notes"

model_name = "BAAI/bge-small-en-v1.5"

model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}

db_exists = (
    Path(DB_PATH).exists()
    and any(Path(DB_PATH).iterdir())
)

# Initialize the embedding model when required
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

if db_exists:
    print("Loading existing ChromaDB...\n\n")

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=DB_PATH,
    )

else:
    print("Database not found. Creating it for the first time...\n\n")

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=DB_PATH,
    )

    print("PDF processed and database created!\n\n")

query = "What is git and github?"

results = vectorstore.similarity_search(query, k=2)

for result in results:
    print(result.page_content)
    print(result.metadata)


retriver = vectorstore.as_retriever()

docs = retriver.invoke("Explain why use git?")

for doc in docs:
    print(doc.page_content)