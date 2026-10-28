from typing import List
import arxiv
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

class ArxivRetriever(BaseRetriever):
    load_max_docs: int = 2

    def _get_relevant_documents(self, query: str) -> List[Document]:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=self.load_max_docs,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        docs = []
        for result in client.results(search):
            metadata = {
                "title": result.title,
                "authors": ", ".join(author.name for author in result.authors),
                "entry_id": result.entry_id,
            }
            docs.append(Document(page_content=result.summary, metadata=metadata))
        return docs

# Instantiate and run via standard invoke() interface
retriever = ArxivRetriever(load_max_docs=2)
docs = retriever.invoke("large language models")

# Print results
for i, doc in enumerate(docs):
    print(f"\nResult {i+1}")
    print("Title:", doc.metadata.get("title"))
    print("Authors:", doc.metadata.get("authors"))
    print("Summary:", doc.page_content[:500])