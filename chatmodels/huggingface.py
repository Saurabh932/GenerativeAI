from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    temperature=0.7,
)

model = ChatHuggingFace(llm=llm)
model.invoke("Hey, Who are you?")
print(model.content)