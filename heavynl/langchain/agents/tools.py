from langchain.tools import BaseTool
from pydantic import BaseModel, Extra, Field

from heavynl.langchain.index import get_heavydb_index
from heavynl.langchain import HeavyDB
from heavynl.langchain.chains import AskHeavyDBMetadataIndexChain


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
Provide a comma-separated list of tables as input, and this tool will output the schema and sample rows for those tables.
This is useful if the human user has supplied the table names as part of the original question.

Example Input: table1, table2, table3
"""

    def _run(self, table_names: str) -> str:
        """Get the schema for tables in a comma-separated list."""
        if "'" in table_names:
            table_names = table_names.replace("'", "")
        return self.db.get_table_info_no_throw([tn.strip() for tn in table_names.strip().split(",")])

    async def _arun(self, table_names: str) -> str:
        raise NotImplementedError("RetrieveTableSchemasTool does not support async")
