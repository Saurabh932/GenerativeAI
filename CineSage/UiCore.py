import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Information Extractor",
    page_icon="📄",
    layout="centered",
)

st.title("📄 Movie Information Extractor")
st.caption("Extract useful information and generate a quick summary.")

# Load model once
@st.cache_resource
def load_model():
    llm = HuggingFacePipeline.from_model_id(
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 512,
            "do_sample": False,
            "repetition_penalty": 1.03,
            "return_full_text": False,
            "clean_up_tokenization_spaces": False,
        },
    )

    return ChatHuggingFace(llm=llm)


model = load_model()

# Extraction prompt
prompt = ChatPromptTemplate.from_template("""
You are an expert information extraction assistant.

Your task is to analyze the provided paragraph about a movie
and extract its useful information accurately and concisely.

Use only the information in the paragraph.
Do not invent facts or use external knowledge.
If a detail is not provided, write "Not mentioned".
If a genre or other detail is inferred rather than explicitly stated,
clearly label it as "Inferred".

MOVIE PARAGRAPH:
{paragraph}

Extract the information using the following format:

MOVIE INFORMATION

1. Movie Title:
2. Release Year:
3. Genre:
4. Director:
5. Cast:
6. Main Characters:
7. Plot:
8. Setting:
9. Main Themes:
10. Composer / Soundtrack:
11. IMDb Rating:
12. Critical or Audience Reception:
13. Notable Features:
14. Awards or Achievements:
15. Other Important Details:

QUICK SUMMARY
Write a concise summary of the paragraph in 2–3 sentences.
Capture what the movie is about and the most important facts.

KEY TAKEAWAYS
Provide up to 3 short bullet points containing the most useful
information from the paragraph.

RULES:
- Extract every relevant detail that is actually available.
- Preserve names, dates, ratings, and other factual details accurately.
- List multiple cast members separately, using their provided names.
- Do not assume that actors' names are character names.
- Keep the plot concise without adding new story details.
- Do not confuse a movie's themes with its genre.
- Avoid repeating the same information unnecessarily.
- Do not add an introduction or commentary about your task.
- If the paragraph does not contain enough information for a field,
  write "Not mentioned".
- Return only the requested output in the format above.
""")

# Paragraph input
paragraph = st.text_area(
    "Enter your paragraph",
    height=200,
    placeholder="Paste a paragraph about a movie here...",
)

# Extract information
if st.button("Extract Information", type="primary", use_container_width=True):
    if not paragraph.strip():
        st.warning("Please enter a paragraph first.")
    else:
        with st.spinner("Extracting information..."):
            final_prompt = prompt.invoke({
                "paragraph": paragraph
            })

            response = model.invoke(final_prompt)

        st.subheader("Extracted Information")
        st.markdown(response.content)