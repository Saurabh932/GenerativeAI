from h11._abnf import chunk_size
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# data = TextLoader("document_loaders/notes.txt", encoding="utf-8")
# data = PyPDFLoader("document_loaders/Git Notes.pdf")
# doc = data.load()

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=100,
# )

# chunks = splitter.split_documents(doc)

template = ChatPromptTemplate.from_messages(
     [
        ("system", "You are an AI that summarizes the text"),
        ("human", "{data}")
     ]
)

llm = HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens":512,
        "do_sample":False,
        "repetition_penalty":1.03,
        "return_full_text":False,
        "clean_up_tokenization_spaces":False
    }
)

model = ChatHuggingFace(llm=llm)
prompt = template.format_messages(data = doc[0].page_content)
result = model.invoke(prompt)

print(result.content)