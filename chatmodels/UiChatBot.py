import streamlit as st
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="💬",
    layout="centered",
)

# UI styling only
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }

    h1 {
        text-align: center;
        color: #f5f7fa;
    }

    .subtitle {
        text-align: center;
        color: #9ca3af;
        margin-bottom: 28px;
    }

    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 12px;
        border: 1px solid #293241;
    }

    [data-testid="stChatInput"] {
        border-color: #374151;
    }
</style>
""", unsafe_allow_html=True)

st.title("💬 AI Chatbot")
st.markdown(
    '<p class="subtitle">Your conversation, your personality.</p>',
    unsafe_allow_html=True,
)

# Choose chatbot mood
mood = st.selectbox(
    "Choose your chatbot mood",
    ["😂 Funny", "😡 Angry", "🥀 Sad"],
)

moods = {
    "😂 Funny": """
You are extremely funny, witty, sarcastic, and playful.
Use clever jokes, unexpected punchlines, dramatic exaggerations,
and intense humor. Make the conversation entertaining and natural.
""",

    "😡 Angry": """
You are intensely angry, fiery, dramatic, and brutally direct.
Use sharp sarcasm, strong opinions, and passionate reactions.
Express frustration theatrically while remaining useful.
Do not threaten or encourage violence.
""",

    "🥀 Sad": """
You are deeply sad, melancholic, emotional, and poetic.
Express yourself with heartfelt intensity, meaningful metaphors,
and reflective thoughts. Keep your responses emotionally deep
while still answering the user's actual questions.
""",
}

content = f"""
You are a conversational AI.

The user may assign you a name or identity during the conversation.
Remember it using the conversation history.

Your current personality is:
{moods[mood]}

Stay in character throughout the conversation.
"""

# Initialize the model once per Streamlit server process
@st.cache_resource
def load_model():
    llm = HuggingFacePipeline.from_model_id(
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        task="text-generation",
        device=0,
        pipeline_kwargs={
            "max_new_tokens": 256,
            "do_sample": False,
            "repetition_penalty": 1.03,
            "return_full_text": False,
            "clean_up_tokenization_spaces": False,
        },
    )

    return ChatHuggingFace(llm=llm)


model = load_model()

# Preserve conversation across Streamlit reruns
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=content)
    ]

# Display existing conversation
for message in st.session_state.messages:
    if isinstance(message, SystemMessage):
        continue

    role = "user" if isinstance(message, HumanMessage) else "assistant"

    with st.chat_message(role):
        st.markdown(message.content)

# Chat input
prompt = st.chat_input("Type your message...")

if prompt:
    # Reset conversation
    if prompt.strip() == "0":
        st.session_state.messages = [
            SystemMessage(content=content)
        ]
        st.rerun()

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append(
        HumanMessage(content=prompt)
    )

    # Generate response using the selected mood
    conversation = [
        SystemMessage(content=content)
    ] + [
        message
        for message in st.session_state.messages
        if not isinstance(message, SystemMessage)
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = model.invoke(conversation)
            st.markdown(response.content)

    # Store assistant response
    st.session_state.messages.append(
        AIMessage(content=response.content)
    )