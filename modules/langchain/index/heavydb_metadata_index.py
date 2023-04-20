from typing import Literal

from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.schema import Document
from langchain.text_splitter import TextSplitter

from modules.langchain.chains import (
    SQLMetadataQuestionTransformerChain,
    AskHeavyDBMetadataIndexChain,
)
from modules.langchain.logging import log_chain_call

search_types = Literal["similarity", "mmr"]


class HeavyDBMetadataIndex(VectorStoreIndexWrapper):
    text_splitter: TextSplitter

    def simple_search_for_table_docs(
        self, search_string: str, search_type: search_types = "similarity", k: int = 5, fetch_k: int = 20
    ) -> list[Document]:
        """Does a simple search for documents that contain text similar to the search string.

        Args:
            search_string: Text to look up documents similar to. The table_name will be taken from the metadata of the documents.
            search_type: Type of search to perform. Defaults to "similarity".
                - similarity: Return docs most similar to query.
                - mmr: Return docs selected using the maximal marginal relevance.
            k: Number of Documents to return. Defaults to 5.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.

        Returns:
            List of table Documents that MAY be relevant to the search string.
        """
        retriever = self.vectorstore.as_retriever(search_type=search_type, search_kwargs={"k": k, "fetch_k": fetch_k})
        return retriever.get_relevant_documents(search_string)

    def simple_search_for_table_names(
        self, search_string: str, search_type: search_types = "similarity", k: int = 5, fetch_k: int = 20
    ) -> list[str]:
        """Does a simple search for documents that contain text similar to the search string. Returns the table names.
        NOTE: This does not use an LLM to determine relevancy, but it is quick.
        You are likely to be returned more tables than you need.

        Args:
            search_string: Text to look up documents similar to. The table_name will be taken from the metadata of the documents.
            search_type: Type of search to perform. Defaults to "similarity".
                - similarity: Return docs most similar to query.
                - mmr: Return docs selected using the maximal marginal relevance.
            k: Number of Documents to return. You want to return at least as many as you expect tables to be returned
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.

        Returns:
            List of table names that MAY be relevant to the search string.
        """
        docs = self.simple_search_for_table_docs(search_string, search_type, k, fetch_k)
        return list(set([doc.metadata["source"] for doc in docs]))

    def ask_about_database(
        self,
        question: str,
        search_type: search_types = "similarity",
        k: int = 6,
        fetch_k: int = 20,
    ) -> dict[str, str]:
        """Ask a question akin to: "Which table(s) are about [topic]?", "Which column of the [table_name] table is used for [thing]?"

        Args:
            search_string: Text to look up documents similar to. The table_name will be taken from the metadata of the documents.
            search_type: Type of search to perform. Defaults to "similarity".
                - similarity: Return docs most similar to query.
                - mmr: Return docs selected using the maximal marginal relevance.
            k: Number of Documents to return. Defaults to 6.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.

        Returns:
            List of dict containing:
              - answer: natural language answer
              - tables: list[str]
        """

        retriever = self.vectorstore.as_retriever(search_type=search_type, search_kwargs={"k": k, "fetch_k": fetch_k})
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

    def ask_using_rephrased_question(self, question: str) -> dict[str, str]:
        """Ask a question by first rephrasing the input question and then using the rephrased question to query the database.

        Args:
            question: Original question to ask the database.

        Returns:
            Dictionary containing:
            - answer: A natural language answer to the question.
            - tables: A list of table names relevant to the answer.
        """
        rephrased_question = self.rephrase_question(question)
        return self.ask_about_database(rephrased_question)
