from h11._abnf import chunk_size
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# Load the PDF
print("Loading the PDF...\n")
data = PyPDFLoader("document_loaders/Git Notes.pdf")
doc = data.load()
print("PDF loaded successfully...\n")


# Split into Chunks
print("Splitting the data ...\n")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

print("Splitting Done, now chunking the data ...\n")
chunks = splitter.split_documents(doc)
print("Chunks created ...\n")

# Creating the Embedding
print("Converting chunks into embeddings...\n")
model_name = "BAAI/bge-small-en-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}

embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
print("Embedding Done...\n")

# Store into Chrome
print("Storing the in ChromaDB...\n")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma-db"
)

