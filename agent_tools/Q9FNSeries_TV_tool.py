from langchain.tools import BaseTool
from typing import Any

class Q9FNSeries_TV_Retriever(BaseTool):
    # MUST annotate these fields
    name: str = "Q9FNSeriesTV_Retriever"
    description: str = " Return top 3 relevant document chunks from Samsung Q9FNSeries TV User Manual."

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
        return self._run(query)
