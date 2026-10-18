from dotenv import load_dotenv

load_dotenv()

'''Gemini'''
# from langchain_google_genai import ChatGoogleGenerativeAI
# model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
# response = model.invoke("Give me a pragraph on Machine Learning and Backend systems")
# print(response.content)



'''Grok'''
from langchain.chat_models import init_chat_model
model = init_chat_model("groq:openai/gpt-oss-120b")
response = model.invoke("What is Cache?")
print(response.content)