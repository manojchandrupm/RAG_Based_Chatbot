from langchain.tools import BaseTool
from typing import Any

class PolicyRetriever(BaseTool):
    # MUST annotate these fields
    name: str = "PolicyRetriever"
    description: str = " Return top 3 relevant document chunks from New India Mediclaim Policy."

    retriever: Any
    llm: Any

    def _run(self, query: str) -> str:
        """Synchronous run (ReAct agent will call this)."""
        docs = self.retriever.invoke(query)
        seen = set()
        unique_texts = []
        for d in docs:
            text = d.page_content.strip()
            if text not in seen:
                seen.add(text)
                unique_texts.append(text)

        combined_text = "\n\n".join(unique_texts)
        return combined_text

    async def _arun(self, query: str) -> str:
        """Async run (not used here)."""
        docs = self.retriever.invoke(query)
        seen = set()
        unique_texts = []
        for d in docs:
            text = d.page_content.strip()
            if text not in seen:
                seen.add(text)
                unique_texts.append(text)

        combined_text = "\n\n".join(unique_texts)
        return combined_text
