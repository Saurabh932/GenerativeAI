from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import  CharacterTextSplitter, TokenTextSplitter

# Character Text Splitting
# splitter = CharacterTextSplitter(
#     separator="",
#     chunk_size = 10,
#     chunk_overlap = 1
# )

# Token Based Splitting
splitter = TokenTextSplitter(
    chunk_size = 100,
    chunk_overlap = 10
)

# data = TextLoader("document_loaders/notes.txt",  encoding="utf-8") # For character text splitter
data = PyPDFLoader("document_loaders/Git Notes.pdf")
doc = data.load()

chunks = splitter.split_documents(doc)
print(len(chunks))  