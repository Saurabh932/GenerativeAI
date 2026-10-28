import hashlib
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

# Paths are relative to this app.py file, so the app works regardless of
# the terminal's current working directory.
PROJECT_DIR = Path(__file__).resolve().parent
CHROMA_DIR = PROJECT_DIR / "chroma-db"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

st.set_page_config(
    page_title="BookChat — Ask your PDF",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1120px; padding-top: 2rem; padding-bottom: 3rem;}
      .hero {padding: 1.5rem 1.6rem; border-radius: 18px;
             background: linear-gradient(120deg, #eef2ff 0%, #f0fdfa 100%);
             border: 1px solid #e0e7ff; margin-bottom: 1.2rem;}
      .hero h1 {margin: 0; font-size: 2rem; color: #172554;}
      .hero p {margin: .5rem 0 0 0; color: #475569;}
      .small-muted {font-size: .9rem; color: #64748b;}
      [data-testid="stChatMessage"] {border-radius: 14px;}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner="Loading the embedding model…")
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource(show_spinner="Loading the local language model… This may take a while the first time.")
def get_chat_model():
    llm = HuggingFacePipeline.from_model_id(
        model_id=LLM_MODEL_NAME,
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


def collection_name_for(pdf_bytes: bytes) -> str:
    # Chroma collection names must be valid and this produces a stable name
    # for the same PDF, so re-uploading the same book reuses its collection.
    digest = hashlib.sha256(pdf_bytes).hexdigest()[:24]
    return f"book_{digest}"


def index_pdf(pdf_bytes: bytes, filename: str):
    """Load a PDF, split it into chunks, embed it, and store it in Chroma."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_bytes)
        temp_path = temp_file.name

    try:
        pages = PyPDFLoader(temp_path).load()
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    if not pages or not any(page.page_content.strip() for page in pages):
        raise ValueError(
            "No selectable text was found in this PDF. If it is a scanned book, "
            "it needs OCR before it can be indexed."
        )

    # Replace the temporary file path with the uploaded filename in metadata.
    for page in pages:
        page.metadata["source"] = filename

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(pages)

    if not chunks:
        raise ValueError("No text chunks could be created from this PDF.")

    embeddings = get_embedding_model()
    collection_name = collection_name_for(pdf_bytes)

    # Chroma persists the collection under CHROMA_DIR. Each PDF has its own
    # collection so books do not get mixed together.
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(CHROMA_DIR),
    )
    return vectorstore, len(pages), len(chunks), collection_name


def get_vectorstore(collection_name: str):
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embedding_model(),
    )


def answer_question(question: str, vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )
    docs = retriever.invoke(question)

    if not docs:
        return "I could not find the answer in the document.", []

    context = "\n\n".join(
        f"[Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant answering questions about an uploaded book. "
                "Use ONLY the supplied context. Do not use outside knowledge or invent facts. "
                "If the answer is not supported by the context, reply exactly: "
                "'I could not find the answer in the document.' "
                "When possible, mention the page number shown in the context.",
            ),
            (
                "human",
                "Book context:\n{context}\n\nQuestion:\n{question}",
            ),
        ]
    )

    model = get_chat_model()
    response = model.invoke(
        prompt.invoke({"context": context, "question": question})
    )
    return response.content, docs


# --------------------------- Sidebar ---------------------------
with st.sidebar:
    st.markdown("## 📚 BookChat")
    st.caption("A local RAG assistant for your PDF books.")
    st.divider()
    st.markdown("### 1 · Upload a book")
    uploaded_pdf = st.file_uploader(
        "Choose a PDF book",
        type=["pdf"],
        accept_multiple_files=False,
        help="Your PDF is processed locally by this app and indexed for question answering.",
    )

    process_clicked = st.button(
        "Process book",
        type="primary",
        use_container_width=True,
        disabled=uploaded_pdf is None,
    )

    if process_clicked and uploaded_pdf is not None:
        try:
            with st.status("Preparing your book…", expanded=True) as status:
                st.write("Reading PDF pages…")
                vectorstore, page_count, chunk_count, collection_name = index_pdf(
                    uploaded_pdf.getvalue(), uploaded_pdf.name
                )
                st.write("Creating embeddings and saving the book index…")
                st.session_state.active_collection = collection_name
                st.session_state.active_book_name = uploaded_pdf.name
                st.session_state.book_page_count = page_count
                st.session_state.book_chunk_count = chunk_count
                st.session_state.messages = []
                status.update(label="Book is ready!", state="complete", expanded=False)
            st.success(f"Indexed {page_count} pages into {chunk_count} chunks.")
        except Exception as exc:
            st.error(f"Could not process this PDF: {exc}")

    st.divider()
    st.markdown("### 2 · Ask questions")
    if st.session_state.get("active_collection"):
        st.success(f"Active book: **{st.session_state.get('active_book_name', 'PDF')}**")
        st.caption(
            f"{st.session_state.get('book_page_count', 0)} pages · "
            f"{st.session_state.get('book_chunk_count', 0)} text chunks"
        )
    else:
        st.info("Upload a PDF and select **Process book** to get started.")

    st.divider()
    st.caption("Models run locally on your machine. The first model load can take time.")

# ----------------------------- Main ----------------------------
st.markdown(
    """
    <div class="hero">
      <h1>Ask your book anything.</h1>
      <p>Upload a PDF, let BookChat index it, and ask questions grounded in its contents.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.get("active_collection"):
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Your personal book assistant")
        st.write(
            "BookChat searches the book for relevant passages and uses a local language "
            "model to draft an answer from those passages."
        )
        st.markdown(
            "- **Upload** a PDF book in the sidebar.\n"
            "- **Process** it to create searchable chunks and embeddings.\n"
            "- **Ask** questions and review the source passages used."
        )
    with right:
        st.info(
            "**Supported input:** text-based PDF files.\n\n"
            "Scanned PDFs that contain only page images need OCR first."
        )
else:
    st.subheader(f"💬 Chat about {st.session_state.get('active_book_name', 'your book')}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("View source passages"):
                    for idx, source in enumerate(message["sources"], start=1):
                        page = source.metadata.get("page", None)
                        page_label = page + 1 if isinstance(page, int) else "Unknown"
                        st.markdown(f"**Source {idx} · Page {page_label}**")
                        st.write(source.page_content)
                        if idx < len(message["sources"]):
                            st.divider()

    question = st.chat_input("Ask a question about the uploaded book…")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching the book and preparing an answer…"):
                    store = get_vectorstore(st.session_state.active_collection)
                    answer, source_docs = answer_question(question, store)
                st.markdown(answer)
                if source_docs:
                    with st.expander("View source passages"):
                        for idx, source in enumerate(source_docs, start=1):
                            page = source.metadata.get("page", None)
                            page_label = page + 1 if isinstance(page, int) else "Unknown"
                            st.markdown(f"**Source {idx} · Page {page_label}**")
                            st.write(source.page_content)
                            if idx < len(source_docs):
                                st.divider()
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": source_docs}
                )
            except Exception as exc:
                error_message = f"I couldn't answer that question because an error occurred: {exc}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )
