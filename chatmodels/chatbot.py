
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 256,
        "do_sample": False,
        "repetition_penalty": 1.03,
        "return_full_text": False,
        "clean_up_tokenization_spaces": False,
    },
)

model = ChatHuggingFace(llm=llm)

content = "You will be assigned a name or title, then act according to it"

# Conversation history starts empty
messages = [
    SystemMessage(content = content)
]

print("-------------Welcome! Type 0 to exit-------------")

while True:
    prompt = input("You: ").strip()
    print()

    if prompt == "0":
        print("Thank you for coming!")
        break

    messages.append(HumanMessage(content=prompt))

    response = model.invoke(messages)

    messages.append(AIMessage(content=response.content))

    print("Bot:", response.content)
    print