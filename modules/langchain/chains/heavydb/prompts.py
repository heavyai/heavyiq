from langchain.prompts.prompt import PromptTemplate

NL_TO_SQL_TEMPLATE = """Given an input question, create a syntactically correct {dialect} query to run to answer the question.
Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.
Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Use GROUP BY if the question can be answered by aggregating over a column.
Exclude null values from the results. Do not use reserved SQL keywords as aliases.
Use the following format:
Question: "Question here"
SQLQuery: "SQL Query to run"
Only use the tables listed below.
{table_info}
Question: {input}"""
NL_TO_SQL_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "dialect"],
    template=NL_TO_SQL_TEMPLATE,
)

NL_TO_SQL_ERROR_TEMPLATE = """Please correct the following {dialect} query:
Use the following format:
Question: "Question here
SQLQuery: "SQL Query to run"
Error: "Error message"
NewSQLQuery: "Fixed SQL Query"
Only use the tables listed below.
{table_info}
Question: {input}
SQLQuery: {sql_cmd}
Error: {error}
"""
NL_TO_SQL_ERROR_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "dialect", "sql_cmd", "error"],
    template=NL_TO_SQL_ERROR_TEMPLATE,
)

ANSWER_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Use the following format:
Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"
Answer: "Final answer here"
Only use the tables listed below.
{table_info}
Question: {input}
SQLQuery: {sql_cmd}
SQLResult: {sql_result}"""
ANSWER_PROMPT = PromptTemplate(
    input_variables=["input", "table_info", "dialect", "sql_cmd", "sql_result"],
    template=ANSWER_TEMPLATE,
)
