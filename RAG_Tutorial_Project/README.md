# BookChat — Streamlit PDF RAG app

## What it does
- Lets a user upload a text-based PDF book in the sidebar.
- Splits the PDF into chunks and creates embeddings using `BAAI/bge-small-en-v1.5`.
- Stores each uploaded book in its own Chroma collection under `chroma-db/`.
- Answers questions using retrieved context and `meta-llama/Llama-3.2-1B-Instruct`.
- Shows the source passages and page numbers used for an answer.

## Setup (Windows PowerShell)
Run these commands from your `D:\GenerativeAI` environment or activate the environment you use for the project:

```powershell
uv pip install -r RAG_Tutorial_Project/requirements-streamlit.txt
streamlit run RAG_Tutorial_Project/app.py
```

If you put these files in a different folder, adjust the paths in the commands.

## Notes
- The first run may download model files and take some time.
- This version uses CPU embeddings and the local Hugging Face pipeline. Model inference can be slow on CPU.
- Input must be a text-based PDF. Scanned/image-only PDFs require OCR first.
- Each PDF uses a separate Chroma collection to avoid mixing books.
- The app only loads and indexes a book when **Process book** is clicked.
- Your existing `main.py` and `create_db.py` are not required to run this UI.
