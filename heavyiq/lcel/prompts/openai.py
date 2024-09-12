# contains plain prompts which might be passed to openai models

ANSWER_TEMPLATE = """Given an input question, first create a syntactically correct SQL query to run, then look at the results of the query and return the answer.
Use the following format:
Question: Question here
SQLQuery: /* step-by-step thought process */ SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here
Question: {input}
Columns Considered:
{columns}
SQLQuery: {sql_cmd}
SQLResult: {sql_result}"""

NL_TO_SQL_TEMPLATE = """Create a syntactically correct SQL query to answer the input question.
Only query relevant columns, avoiding SELECT * for any table.
Only create one query.
These functions DO NOT EXIST: STRING_AGG, GROUP_CONCAT
Use only existing column names from the schema description, ensuring they are from the correct table.
Exclude null values from the results. DO NOT use reserved SQL keywords as aliases.
DO NOT use double colon cast operators ("::"); instead use the CAST function if needed.
Explain your thinking step-by-step in a block comment before the query. Provide your response using the format below:
Question: [QUESTION]
SQLQuery: /* step-by-step reasoning */ [SINGLE SQL QUERY]
Only use the tables listed below. Some of the tables may not be relevant.
{table_info}
Question: {input}
SQLQuery:"""

NL_TO_SQL_ERROR_TEMPLATE = """Correct the given SQL query:
If a function signature does not exist, do not use it. Reformulate the query to not use that function signature.
DO NOT USE STRING_AGG, GROUP_CONCAT functions.
Exclude null values from the results. Do not use reserved SQL keywords as aliases.
Do not use double colon cast operators ("::"); instead use the CAST function if needed.
Use the following format:
Question: [QUESTION]
SQLQuery: /* step-by-step thought process */ [SQL QUERY]
Error: [ERROR MESSAGE]
NewSQLQuery: /* new step-by-step thought process */ [FIXED SQL QUERY]
Only use the tables listed below.
{table_info}
Question: {input}
SQLQuery: {sql_cmd}
Error: {error}
NewSQLQuery:
"""


NL_TO_TABLES_TEMPLATE = """You are an expert data analyst adept at analyzing user questions to determine the SQL tables required to generate an answer.
You have access to the following relational tables, with schemas below.

{table_info}

Emit each table name along with a 1 if the table is required to answer the following user question, or a 0 if the table is not required.

Question: {input}
Tables:"""
