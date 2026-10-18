from torch import embedding
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

model_name = "BAAI/bge-small-en-v1.5"
model_kwargs = {"device":"cpu"}
encode_kwargs = {"normalize_embedding": True}

embeddings = HuggingFaceEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    # encode_kwargs = encode_kwargs
)

query_text = "What is LangChain?"
vector = embeddings.embed_query(query_text)

print(f"Embedding dimensions: {len(vector)}")
print(vector)

documents = [
     "LangChain is a framework designed to simplify the creation of applications using LLMs.",
     "Hugging Face provides open-source tools to build, train and deploy machine learning models."
]

doc_embeddings = embeddings.embed_documents(documents)

print(f"Successfully generated {len(doc_embeddings)} document vectors.")
print(doc_embeddings)
