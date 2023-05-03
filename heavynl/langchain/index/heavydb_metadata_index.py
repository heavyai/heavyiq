from typing import Literal, Optional

from langchain.vectorstores import Chroma
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.schema import Document, BaseRetriever
from langchain.text_splitter import TextSplitter

from heavynl.langchain.chains import (
    SQLMetadataQuestionTransformerChain,
    AskHeavyDBMetadataIndexChain,
)
from heavynl.langchain.logging import log_chain_call
from .utils import read_table_documents, apply_retriever_filter


search_types = Literal["similarity", "mmr"]


class HeavyDBMetadataIndex(VectorStoreIndexWrapper):
    text_splitter: TextSplitter
    vectorstore: Chroma

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

    def ask_about_database(
        self,
        question: str,
        allowable_tables: Optional[list[str]] = None,
        **kwargs,
    ) -> dict[str, str]:
        """Ask a question akin to: "Which table(s) are about [topic]?", "Which column of the [table_name] table is used for [thing]?"

        Args:
            question: Text to look up documents similar to. The table_name will be taken from the metadata of the documents.
            allowable_tables: List of table names to search. If None, all tables will be searched.

        Returns:
            List of dict containing:
              - answer: natural language answer
              - tables: list[str]
        """
        retriever = self.as_retriever(allowable_tables=allowable_tables, **kwargs)
        chain = AskHeavyDBMetadataIndexChain.create(retriever=retriever)
        res: dict[str, str] = log_chain_call(chain, question, "", chain_name="ask_heavydb_metadata_index")
        if res[chain.answer_key] == chain.no_results_answer:
            # consider a fallback to a simple search
            raise Exception("No relevant tables found for query: " + question)
        return {
            "answer": res[chain.answer_key],
            "tables": res[chain.tables_answer_key],
        }

    def rephrase_question(self, question: str) -> str:
        """Rephrase a question to be more like a question that this index can answer.

        Args:
            question: Question to rephrase.

        Returns:
            Rephrased question.
        """
        chain = SQLMetadataQuestionTransformerChain()
        return log_chain_call(chain, {chain.input_key: question}, "")[chain.output_key]

    def ask_using_rephrased_question(self, question: str, **kwargs) -> dict[str, str]:
        """Ask a question by first rephrasing the input question and then using the rephrased question to query the database.

        Args:
            question: Original question to ask the database.

        Returns:
            Dictionary containing:
            - answer: A natural language answer to the question.
            - tables: A list of table names relevant to the answer.
        """
        rephrased_question = self.rephrase_question(question)
        return self.ask_about_database(rephrased_question, **kwargs)

    def reindex_table_document(self, table_name: str) -> None:
        docs = list(read_table_documents(include=[table_name]))
        if len(docs) == 0:
            raise Exception(f"No document found for table {table_name}")
        self.vectorstore._collection.delete(where={"source": table_name})
        sub_docs = self.text_splitter.split_documents(docs)
        self.vectorstore.add_documents(sub_docs)

    def as_retriever(
        self,
        search_type: search_types = "similarity",
        k: int = 6,
        fetch_k: int = 20,
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
        search_kwargs = {"k": k, "fetch_k": fetch_k}
        search_kwargs = apply_retriever_filter(search_kwargs, allowable_tables)
        return self.vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
