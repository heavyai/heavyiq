from typing import Optional

from chromadb.api.types import Where
from fastapi.concurrency import run_in_threadpool
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

from heavyiq.langchain.heavydb import HeavyDB

from ..utils import HeavyIQIndexWrapper, SearchType
from .generate_table_documents import agenerate_table_document
from .utils import apply_retriever_filter, aread_table_documents, read_table_documents


class HeavyDBMetadataIndex(HeavyIQIndexWrapper):
    def simple_search_for_table_docs(
        self, search_string: str, allowable_tables: Optional[list[str]] = None, **kwargs
    ) -> list[Document]:
        """Does a simple search for documents that contain text similar to the search string.

        Args:
            search_string: Text to look up documents similar to. The table_name will be taken from the metadata of the documents.
            allowable_tables: List of table names to search. If None, all tables will be searched.

        Returns:
            List of table Documents that MAY be relevant to the search string.
        """
        retriever = self.as_retriever(allowable_tables=allowable_tables, **kwargs)
        return retriever.get_relevant_documents(search_string)

    def simple_search_for_table_names(
        self, search_string: str, allowable_tables: Optional[list[str]] = None, **kwargs
    ) -> list[str]:
        """Does a simple search for documents that contain text similar to the search string. Returns the table names.
        NOTE: This does not use an LLM to determine relevancy, but it is quick.
        You are likely to be returned more tables than you need.

        Args:
            search_string: Text to look up documents similar to. The table_name will be taken from the metadata of the documents.
            allowable_tables: List of table names to search. If None, all tables will be searched.

        Returns:
            List of table names that MAY be relevant to the search string.
        """
        docs = self.simple_search_for_table_docs(search_string, allowable_tables, **kwargs)
        return list(set([doc.metadata["source"] for doc in docs]))

    async def asimple_search_for_table_docs(
        self, search_string: str, allowable_tables: Optional[list[str]] = None, **kwargs
    ) -> list[Document]:
        """
        Search for table docs on the index asynchornously.
        """
        retriever = self.as_retriever(allowable_tables=allowable_tables, **kwargs)
        return await retriever.aget_relevant_documents(search_string)

    async def asimple_search_for_table_names(
        self, search_string: str, allowable_tables: Optional[list[str]] = None, **kwargs
    ) -> list[str]:
        """
        Search for table names on the index asynchornously.
        """
        docs = await self.asimple_search_for_table_docs(search_string, allowable_tables, **kwargs)
        return list(set([doc.metadata["source"] for doc in docs]))

    def reindex_table_document(self, table_name: str) -> None:
        docs = list(read_table_documents(include=[table_name]))
        if len(docs) == 0:
            raise Exception(f"No document found for table {table_name}")
        where: Where = {"source": table_name}
        self.vectorstore._collection.delete(where=where)
        sub_docs = self.text_splitter.split_documents(docs)
        self.vectorstore.add_documents(sub_docs)

    async def agenerate_and_reindex_table_document(self, heavydb: HeavyDB, table: str) -> None:
        """
        Coroutine responsible for generating a document for a table, regardless of its existence,
        and subsequently reindexing the document on ChromaDB VectorStore.
        """
        await agenerate_table_document(heavydb=heavydb, table=table)
        docs = [doc async for doc in aread_table_documents(include=[table])]
        if len(docs) == 0:
            raise Exception(f"No document found for table {table}")
        where: Where = {"source": table}
        await run_in_threadpool(self.vectorstore._collection.delete, where=where)
        sub_docs = await self.text_splitter.atransform_documents(docs)
        await self.vectorstore.aadd_documents(sub_docs)  # type: ignore

    def as_retriever(
        self,
        search_type: SearchType = SearchType.SIMILARITY,
        k: int = 6,
        fetch_k: int = 20,
        score_threshold: float = 0.8,
        lambda_mult: float = 0.5,
        allowable_tables: Optional[list[str]] = None,
    ) -> BaseRetriever:
        """
        Args:
            search_type: Type of search to perform. Defaults to "similarity".
                - similarity: Return docs most similar to query.
                - mmr: Return docs selected using the maximal marginal relevance.
            k: Number of Documents to return. Defaults to 6.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm. Defaults to 20.
            allowable_tables: List of table names to search. If None, all tables will be searched.

        Returns:
            Retriever that can be used to search the index.
        """
        # latest chroma version won't suport fetch_k, so drop it
        search_kwargs = {"k": k}
        search_kwargs = apply_retriever_filter(search_kwargs, allowable_tables)
        if search_type == SearchType.SIMILARITY_SCORE_THRESHOLD:
            search_kwargs["score_threshold"] = score_threshold  # type: ignore
        elif search_type == SearchType.MMR:
            search_kwargs["lambda_mult"] = lambda_mult  # type: ignore

        return self.vectorstore.as_retriever(search_type=search_type.value, search_kwargs=search_kwargs)
