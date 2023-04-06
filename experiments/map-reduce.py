from typing import Optional, Any

from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.summarize import map_reduce_prompt
from langchain.prompts.base import BasePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import BaseLanguageModel
from pydantic import BaseModel, Field, validator

from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

from modules.langchain.heavydb import HeavyDB


def load_map_reduce_chain(
    llm: BaseLanguageModel,
    map_prompt: BasePromptTemplate = map_reduce_prompt.PROMPT,
    combine_prompt: BasePromptTemplate = map_reduce_prompt.PROMPT,
    combine_document_variable_name: str = "text",
    map_reduce_document_variable_name: str = "text",
    collapse_prompt: Optional[BasePromptTemplate] = None,
    reduce_llm: Optional[BaseLanguageModel] = None,
    collapse_llm: Optional[BaseLanguageModel] = None,
    verbose: bool = True,
    **kwargs: Any,
) -> MapReduceDocumentsChain:
    map_chain = LLMChain(llm=llm, prompt=map_prompt, verbose=verbose)
    _reduce_llm = reduce_llm or llm
    reduce_chain = LLMChain(llm=_reduce_llm, prompt=combine_prompt, verbose=verbose)
    # TODO: document prompt
    combine_document_chain = StuffDocumentsChain(
        llm_chain=reduce_chain,
        document_variable_name=combine_document_variable_name,
        verbose=verbose,
    )
    if collapse_prompt is None:
        collapse_chain = None
        if collapse_llm is not None:
            raise ValueError(
                "collapse_llm provided, but collapse_prompt was not: please "
                "provide one or stop providing collapse_llm."
            )
    else:
        _collapse_llm = collapse_llm or llm
        collapse_chain = StuffDocumentsChain(
            llm_chain=LLMChain(
                llm=_collapse_llm,
                prompt=collapse_prompt,
                verbose=verbose,
            ),
            document_variable_name=combine_document_variable_name,
        )
    return MapReduceDocumentsChain(
        llm_chain=map_chain,
        combine_document_chain=combine_document_chain,
        document_variable_name=map_reduce_document_variable_name,
        collapse_document_chain=collapse_chain,
        verbose=verbose,
        **kwargs,
    )


class MapResponse(BaseModel):
    table_name: str = Field(description="The name of the table.")
    columns: list[str] = Field(description="The columns that are relevant to the query, if any.")
    is_relevant: bool = Field(description="Whether the table is relevant to the query.")


parser = PydanticOutputParser(pydantic_object=MapResponse)

MAP_QUERY = """Provided a SQL table schema and a query, determine if the SQL database is relevant to the query.
Do not attempt to answer the query. Ensure you return the name of the table and columns of interest, if any. If it is not relevant, say so:

Example Query: How many entries are there in the firewall log table?

Example SQL Table Schema:
CREATE TABLE sfmta_buses (
  REV SMALLINT,
  REPORT_TIME TIMESTAMP(0),
  VEHICLE_TAG SMALLINT,
  LONGITUDE FLOAT,
  LATITUDE FLOAT,
  SPEED FLOAT,
  HEADING FLOAT,
  TRAIN_ASSIGNMENT SMALLINT,
  PREDICTABLE SMALLINT)
WITH (MAX_CHUNK_SIZE=1073741824);

EXAMPLE OUTPUT: sfmta_buses is not relevant to the query.

Query: {query}

SQL Table Schema:
{schema}

OUTPUT:"""

COMBINE_QUERY = """Ignore any irrelevant tables. Combine the relevant tables a single summary:

SUMMARIES:
{summaries}

RELEVANT TABLE NAME AND COLUMNS:"""


def main() -> None:
    db = HeavyDB.from_env()
    llm = ChatOpenAI(temperature=0.0, client=None, model_name="gpt-3.5-turbo")
    query = "What actor appeared in the most action movies in the 1990s?"
    as_relates_to_query_map = PromptTemplate(
        template=MAP_QUERY,
        input_variables=["schema"],
        partial_variables={"query": query},
    )
    docs: list[Document] = []
    for table in db.get_usable_table_names():
        docs.append(Document(page_content=db.get_table_schema(table), metadata={"source": table}))
    as_relates_to_query_combine = PromptTemplate(template=COMBINE_QUERY, input_variables=["summaries"])

    map_reduce_documents_chain = load_map_reduce_chain(
        llm=llm,
        map_prompt=as_relates_to_query_map,
        combine_prompt=as_relates_to_query_combine,
        map_reduce_document_variable_name="schema",
        combine_document_variable_name="summaries",
    )

    res = map_reduce_documents_chain.combine_docs(docs)
    print(res)


if __name__ == "__main__":
    main()
