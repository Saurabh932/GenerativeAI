from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace,  HuggingFacePipeline
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

data = WebBaseLoader("https://www.apple.com/in/macbook-pro/")
doc = data.load()

# print(doc)
# print(len(doc))
# print(doc[0].page_content)

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
prompt = template.format_messages(data = doc[0].page_content) # Sending just page 1
result = model.invoke(prompt)

print(result.content)