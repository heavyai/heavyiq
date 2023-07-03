from langchain.tools import BaseTool
from pydantic import BaseModel, Extra, Field
from typing import Optional, Type

from heavynl.langchain.index import get_heavydb_index
from heavynl.langchain import HeavyDB
from heavynl.langchain.chains import AskHeavyDBMetadataIndexChain
from heavynl.logging_utils import heavynl_logger as logger


class BaseHeavyDBTool(BaseModel):
    """Base tool for interacting with a HeavyDB database."""

    db: HeavyDB = Field(exclude=True)

    # Override BaseTool.Config to appease mypy
    # See https://github.com/pydantic/pydantic/issues/4173
    class Config(BaseTool.Config):
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True
        extra = Extra.forbid


class RetrieveRelevantSchemasTool(BaseHeavyDBTool, BaseTool):
    """This tool uses a HeavyDB instance and a metadata index to find the relevant tables for a given query.
    It returns the schemas and sample rows for the relevant tables.

    Attributes:
    - name (str): The name of the tool, "query_for_relevant_tables".
    - description (str): A description of the tool and its usage.

    Methods:
    - _run(query: str) -> str: Executes the tool with the given query and returns the relevant table schemas.
    - _arun(query: str) -> str: Not implemented for this tool."""

    name = "query_for_relevant_tables"
    description = """
    If you can't tell which tables are relevant to the query, use this tool to retrieve the schemas of tables relevant to a query.

The output will be schemas for the tables that are relevant to the prompt.

Example Input: 'Which table contains information on [topic]? What are the relevant columns for [foo] and [bar]?'"""

    def _run(self, query: str) -> str:
        try:
            retriever = get_heavydb_index().as_retriever(k=5, allowable_tables=list(self.db.get_usable_table_names()))
            chain = AskHeavyDBMetadataIndexChain.create(retriever=retriever)
            res: dict[str, str] = chain(query)
            if res["answer"] == chain.no_results_answer:
                return "Error: No relevant tables found for provided query. Feel free to rephrase and try again."
            if len(res["tables"]) > 2:
                table_info = self.db.get_table_info(res["tables"], include_samples=False, include_top_k=False)  # type: ignore
            else:
                table_info = self.db.get_table_info(res["tables"])  # type: ignore
            return f"{res['answer']}\n{table_info}"
        except Exception as e:
            print(e)
            return "Error: LLM could not find a relevant table"

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("RetrieveRelevantSchemasTool does not support async")


class QueryHeavyDBTool(BaseHeavyDBTool, BaseTool):
    """This tool uses a HeavyDB instance to execute SQL queries and return the results or an error message.

    Attributes:
    - name (str): The name of the tool, "run_sql_query".
    - description (str): A description of the tool and its usage.

    Methods:
    - _run(query: str) -> str: Executes the SQL query with the given HeavyDB instance and returns the results or an error message.
    - _arun(query: str) -> str: Not implemented for this tool."""

    name = "run_sql_query"
    description = """
    ONLY USE WITH KNOWN TABLE SCHEMAS.
    DONT RUN THE QUERY IF YOU'RE NOT SURE ABOUT THE TABLE SCHEMA.
    This tool is for querying HeavyDB databases.

Input an accurate SQL query having semicolon at the end for output. Apply query limits. Following are the sample action input queries:

SELECT COUNT(*) FROM usa_counties WHERE name LIKE 'A%';
SELECT name FROM usa_counties WHERE name LIKE 'A%';

If an error occurs, revise and retry. Be familiar with table schemas and avoid common mistakes, such as:

- NULL values with NOT IN
- Using UNION instead of UNION ALL
- BETWEEN for exclusive ranges
- Data type mismatch
- Incorrectly quoting identifiers
- Wrong number of arguments in functions
- Incorrect data type casting
- Improper join columns
- Reserved SQL keywords as aliases
- Using an ORDER BY clause without applying a NULLS LAST qualifier"""

    def _run(self, query: str) -> str:
        """Execute the query, return the results or an error message."""
        try:
            self.db.validate_query(query)  # validate the query before running it
            result = self.db.run_no_throw(query)
            return result
        except Exception as e:
            return f"Error: {e}"

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("QueryHeavyDBTool does not support async")


class RetrieveTableSchemasTool(BaseHeavyDBTool, BaseTool):
    """This tool uses a HeavyDB instance to get the schema and sample rows for the provided table names.

    Attributes:
    - name (str): The name of the tool, "retrieve_table_schemas".
    - description (str): A description of the tool and its usage.

    Methods:
    - _run(table_names: str) -> str: Retrieves the schema for tables in a comma-separated list of table names.
    - _arun(table_names: str) -> str: Not implemented for this tool."""

    name = "retrieve_table_schemas"
    description = """
By providing a comma-separated list of table names as input, this tool will output the schema and sample rows for those tables.
This is useful if the human user has supplied the table names as part of the original question.

Example Input: table1, table2, table3
"""

    def _run(self, table_names: str) -> str:
        """Get the schema for tables in a comma-separated list."""
        if "'" in table_names:
            table_names = table_names.replace("'", "")
        return self.db.get_table_info_no_throw([tn.strip() for tn in table_names.strip().split(",")])

    def _arun(self, table_names: str) -> str:
        raise NotImplementedError("RetrieveTableSchemasTool does not support async")


class NewRetrieveRelevantSchemasToolInput(BaseModel):
    """Input for table schema identification."""

    question: str = Field(..., description="Database related question.")


class NewRetrieveRelevantSchemasTool(BaseHeavyDBTool, BaseTool):
    name = "identify_table_schema"
    description = """
        Useful for when you need to find out the tables which are relevant to the asked NL question.
        If you're not sure about the relevant table name, then check for any previously referred table.
        If there's any, then try this "retrieve_table_schemas" tool to get schema information for that table else
        go ahead and use this tool.
        You should input the common database related NL questions which might be asked by a human.
        Run this tool only when the table names are not specified on the question. If the table names are provided then make use of "retrieve_table_schemas" tool.
    """

    def _run(self, question: str) -> str:
        try:
            retriever = get_heavydb_index().as_retriever(k=5, allowable_tables=list(self.db.get_usable_table_names()))
            chain = AskHeavyDBMetadataIndexChain.create(retriever=retriever)
            res: dict[str, str] = chain(question)
            if res["answer"] == chain.no_results_answer:
                return "Error: No relevant tables found for provided question. Feel free to rephrase and try again."
            if len(res["tables"]) > 2:
                table_info = self.db.get_table_info(res["tables"], include_samples=False, include_top_k=False)  # type: ignore
            else:
                table_info = self.db.get_table_info(res["tables"])  # type: ignore
            return f"{res['answer']}\n{table_info}"
        except Exception as exc:
            logger.exception("Exception occurs: %s", exc)
            return "Error: LLM could not find a relevant table"

    def _arun(self, question: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = NewRetrieveRelevantSchemasToolInput


class NewQueryHeavyDBToolInput(BaseModel):
    """Input for generating query results"""

    query: str = Field(..., description="A valid ANSI SQL query with query limits.")


class NewQueryHeavyDBTool(BaseHeavyDBTool, BaseTool):
    name = "run_sql_query"
    description = """
        ONLY USE WITH KNOWN TABLE SCHEMAS.
        Useful for when you need to find out results for a database query. You should input a valid ANSI SQL query with query limits.
        This tool is for querying HeavyDB databases.

        Be familiar with table schemas and avoid common mistakes, such as:
            - NULL values with NOT IN
            - Using UNION instead of UNION ALL
            - BETWEEN for exclusive ranges
            - Data type mismatch
            - Incorrectly quoting identifiers
            - Wrong number of arguments in functions
            - Incorrect data type casting
            - Improper join columns
            - Reserved SQL keywords as aliases
            - Using an ORDER BY clause without applying a NULLS LAST qualifier
        If an error occurs, revise and retry.

        Use the output of this tool to decide what would be the final answer.
    """

    def _run(self, query: str) -> str:
        """Execute the query, return the results or an error message."""
        try:
            self.db.validate_query(query)  # validate the query before running it
            result = self.db.run_no_throw(query)
            return result
        except Exception as exc:
            return f"Error: {exc}"

    def _arun(self, query: str) -> str:
        raise NotImplementedError("QueryHeavyDBTool does not support async")

    args_schema: Optional[Type[BaseModel]] = NewQueryHeavyDBToolInput


class NewRetrieveTableSchemasToolInput(BaseModel):
    table_names: str = Field(
        ...,
        description="single or list of comma seperated table names for which we need to find database table schema.",
    )


class NewRetrieveTableSchemasTool(BaseHeavyDBTool, BaseTool):
    name = "retrieve_table_schemas"
    description = """
    Useful for when you have a single or list of tables names and you want to find out database schema for those tables along with the top k rows.
    By providing a comma-separated list of table names as input, this tool will output the schema and sample rows for those tables.
    This is useful if the human user has supplied the table names as part of the original question.
    Ouput should be list of table schemas.

    Example Input: table1, table2, table3 get the count of each table
    """

    def _run(self, table_names: str) -> str:
        """Get the schema for tables in a comma-separated list."""
        if "'" in table_names:
            table_names = table_names.replace("'", "")
        return self.db.get_table_info_no_throw([tn.strip() for tn in table_names.strip().split(",")])

    def _arun(self, table_names: str) -> str:
        raise NotImplementedError("QueryHeavyDBTool does not support async")

    args_schema: Optional[Type[BaseModel]] = NewRetrieveTableSchemasToolInput
