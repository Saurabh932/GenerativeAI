from langchain_community.document_loaders import TextLoader

data = TextLoader("document_loaders/notes.txt", encoding="utf-8")
doc = data.load()
print(doc[0].page_content)