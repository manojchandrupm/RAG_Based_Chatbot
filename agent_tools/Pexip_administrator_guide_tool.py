from langchain.tools import BaseTool
from typing import Any

class Pexip_Administrator_Guide_Retriever(BaseTool):
    # MUST annotate these fields
    name: str = "Pexip_Administrator_Guide_Retriever"
    description: str = (" Return top 3 relevant document chunks from Hi Pexip Administrator Guide User Manual."
                        "Searches and returns relevant information from the Pexip Administration Guide.")

    retriever: Any
    llm: Any

    def _run(self, query: str) -> str:
        """Synchronous run (ReAct agent will call this)."""
        docs = self.retriever.get_relevant_documents(query)
        seen = set()
        unique_texts = []
        for d in docs:
            text = d.metadata["full_text"].strip()
            if text not in seen:
                seen.add(text)
                unique_texts.append(text)

        combined_text = "\n\n".join(unique_texts)
        return combined_text

    async def _arun(self, query: str) -> str:
        """Async run (not used here)."""
        return self._run(query)
