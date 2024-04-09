DEFAULT_NL_TO_SQL_PROMPT = """<|sql prompt|>
You are an experienced data analyst adept at writing SQL queries to answer user questions.

You have access to the following relational tables, with schemas below.

{table_info}

Write a SQL query to answer the following question:
{input}
<|sql answer|>
"""

DEFAULT_NL_TO_SQL_ERROR_PROMPT = """<|sql error prompt|>
You generated a SQL query that generated an exception when executed in the HeavyDB database.

You have access to the following relation tables, with schemas below.

{table_info}

In attempting to answer the following user question:

{input},
you generated the following SQL query:
{sql_cmd}
, which failed to run in the HeavyDB database, generating the following error:
{error}

Please alter the query to run without error in HeavyDB:
<|sql error answer|>
"""

DEFAULT_SQL_TO_ANSWER_PROMPT = """<|english prompt|>
The user asked the following question:
{input}

By considering the following columns:
{columns}

To answer the question, the following SQL query was generated:
{sql_cmd}

The following results were returned:
{sql_result}

Now explain the results in English, referencing the question and the SQL query as needed.
<|english answer|>
"""

DEFAULT_NL_TO_TABLES_PROMPT = """<|table prompt|>
You are an experienced data analyst adept at analyzing user questions to determine the SQL tables required to generate an answer.
You have access to the following relational tables, with schemas below.

{table_info}

Emit each table name along with a 1 if the table is required to answer the following user question, or a 0 if the table is not required.

Question: {input}

<|table answer|>
"""

DEFAULT_TABLES_TO_QUESTIONS_PROMPT = """<|table question prompt|>
You are an experienced data analyst adept at asking compelling questions of your data.
You have access to the following relational tables, with schemas below.

{table_info}

Write a compelling question to ask of the above data:
<|table question answer|>
"""

DEFAULT_INSTRUCT_PROMPT = """<|prompt|>
{question}
<|answer|>
"""
