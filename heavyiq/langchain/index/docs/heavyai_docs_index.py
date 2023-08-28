from typing import Any

from pydantic import BaseModel

from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain

from heavyiq.langchain.llms import get_llm_by_type, LLMType
from heavyiq.langchain.logging import log_chain_call

from .utils import get_latest_heavyai_docs
from ..utils import HeavyIQIndexWrapper


class AnswerWithSources(BaseModel):
    question: str
    answer: str
    sources: list[str]
    feedback_id: str


class HeavyAIDocsIndex(HeavyIQIndexWrapper):
    """Index for the Heavy.AI Immerse documentation."""

    def ask_documentation(
        self,
        question: str,
        retriever_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> AnswerWithSources:
        llm = get_llm_by_type(LLMType.SQL_TO_ANSWER, temperature=0.0)
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm, retriever=self.vectorstore.as_retriever(**retriever_kwargs), **kwargs
        )
        res = log_chain_call(chain, question, "", chain_name="ask-docs")
        feedback_id = str(res["__run"].run_id) if "__run" in res else ""

        return AnswerWithSources(
            question=question,
            answer=res["answer"].strip(),
            sources=[source.strip() for source in res["sources"].split(", ")],
            feedback_id=feedback_id,
        )

    def update_docs(self):
        """Removes all documentation from the vector store and redownloads/reindexes the newest immerse docs on the HEAVY.AI site."""
        documents = get_latest_heavyai_docs()
        self.vectorstore._collection.delete()
        self.vectorstore.add_documents(documents)
