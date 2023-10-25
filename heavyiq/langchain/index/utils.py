from enum import Enum

from langchain.llms.openai import OpenAI
from typing import Any
from langchain.schema import BaseRetriever
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.schema.language_model import BaseLanguageModel
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.text_splitter import TextSplitter

from heavyiq.config import get_config


def get_vectorstore_index_creator(persist_directory: str) -> VectorstoreIndexCreator:
    """
    Get a VectorstoreIndexCreator instance.
    :param persist_directory: Path to the directory where the vector store index should be persisted.
    :return: A VectorstoreIndexCreator instance.
    """
    config = get_config()
    huggingface_model_name = config.huggingface_embed_model
    return VectorstoreIndexCreator(
        vectorstore_kwargs={"persist_directory": persist_directory},
        embedding=HuggingFaceEmbeddings(model_name=huggingface_model_name),
    )


class SearchType(Enum):
    SIMILARITY = "similarity"
    MMR = "mmr"


class HeavyIQIndexWrapper(VectorStoreIndexWrapper):
    text_splitter: TextSplitter
    vectorstore: Chroma

    def as_retriever(
        self,
        search_type: SearchType = SearchType.SIMILARITY,
        k: int = 6,
        fetch_k: int = 20,
    ) -> BaseRetriever:
        """
        Args:
            search_type: Type of search to perform. Defaults to "similarity".
                - similarity: Return docs most similar to query.
                - mmr: Return docs selected using the maximal marginal relevance.
            k: Number of Documents to return. Defaults to 6.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm. Defaults to 20.

        Returns:
            Retriever that can be used to search the index.
        """
        search_kwargs = {"k": k, "fetch_k": fetch_k}
        return self.vectorstore.as_retriever(search_type=search_type.value, search_kwargs=search_kwargs)

    async def aquery(
        self,
        question: str,
        llm: BaseLanguageModel | None = None,
        retriever_kwargs: dict[str, Any] | None = None,
        **kwargs: Any
    ) -> str:
        """Query the vectorstore asynchronously."""
        llm = llm or OpenAI(temperature=0)
        retriever_kwargs = retriever_kwargs or {}
        chain = RetrievalQA.from_chain_type(llm, retriever=self.vectorstore.as_retriever(**retriever_kwargs), **kwargs)
        return await chain.arun(question)

    async def aquery_with_sources(
        self,
        question: str,
        llm: BaseLanguageModel | None = None,
        retriever_kwargs: dict[str, Any] | None = None,
        **kwargs: Any
    ) -> dict:
        """Query the vectorstore and get back sources asynchronously."""
        llm = llm or OpenAI(temperature=0)
        retriever_kwargs = retriever_kwargs or {}
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm, retriever=self.vectorstore.as_retriever(**retriever_kwargs), **kwargs
        )
        return await chain.acall({chain.question_key: question})
