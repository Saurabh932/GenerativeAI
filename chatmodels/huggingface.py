
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    provider="featherless-ai",
    temperature=0.7,
    max_new_tokens=128,
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("Hey, who are you?")
print(response.content)