from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens":512,
        "do_sample":False,
        "repetition_penalty":1.03
    }
)

chat_model = ChatHuggingFace(llm=llm)

result = chat_model.invoke("What is alien according to you?")
print(result.content)