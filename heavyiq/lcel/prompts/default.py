DEFAULT_NL_TO_SQL_PROMPT = (
    "<|sql prompt|>\n",
    """You are an expert data analyst adept at writing SQL queries to answer user questions.

You have access to the following relational tables, with schemas below.

Alongside each text column, in parentheses "()" you will see the top 3 values followed by "..." for columns with more than 5 distinct values, or the top 5 values otherwise. For timestamp and date columns you will see the min/max range of the column.

{table_info}

{relevant_info}
Write a SQL query to answer the following question:
{input}""",
    "\n<|sql answer|>\n",
)


DEFAULT_NL_TO_SQL_ERROR_PROMPT = (
    "<|sql error prompt|>\n",
    """You generated a SQL query that generated an exception when executed in the HeavyDB database.

You have access to the following relation tables, with schemas below.

Alongside each text column, in parentheses "()" you will see the top 3 values followed by "..." for columns with more than 5 distinct values, or the top 5 values otherwise. For timestamp and date columns you will see the min/max range of the column.

{table_info}

In attempting to answer the following user question:

{input},
you generated the following SQL query:
{sql_cmd}
, which failed to run in the HeavyDB database, generating the following error:
{error}

{relevant_info}
Please alter the query to run without error in HeavyDB:""",
    "\n<|sql error answer|>\n",
)

DEFAULT_SQL_TO_ANSWER_PROMPT = (
    "<|english prompt|>\n",
    """The user asked the following question:
{input}

By considering the following columns:
{columns}

To answer the question, the following SQL query was generated:
{sql_cmd}

The following results were returned:
{sql_result}

Now explain the results in English, referencing the question and the SQL query as needed.""",
    "\n<|english answer|>\n",
)

DEFAULT_NL_TO_TABLES_PROMPT = (
    "<|table prompt|>\n",
    """You are an expert data analyst adept at analyzing user questions to determine the SQL tables required to generate an answer.

You have access to the following relational tables, with schemas below.

Alongside each text column, in parentheses "()" you will see the top 3 values followed by "..." for columns with more than 5 distinct values, or the top 5 values otherwise. For timestamp and date columns you will see the min/max range of the column.

{table_info}

{relevant_info}
Emit each table name along with a 1 if the table is required to answer the following user question, or a 0 if the table is not required.

Question: {input}
""",
    "\n<|table answer|>\n",
)

DEFAULT_TABLES_TO_QUESTIONS_PROMPT = (
    "<|table question prompt|>\n",
    """You are an expert data analyst adept at asking compelling questions of your data.

You have access to the following relational tables, with schemas below.

Alongside each text column, in parentheses "()" you will see the top 3 values followed by "..." for columns with more than 5 distinct values, or the top 5 values otherwise. For timestamp and date columns you will see the min/max range of the column.

{table_info}

Write a compelling question to ask of the above data:""",
    """\n<|table question answer|>\n""",
)

DEFAULT_INSTRUCT_PROMPT = ("<|prompt|>\n", "{question}", "\n<|answer|>\n")


DEFAULT_NL_TO_SQL_COT_PROMPT = (
    "<|cot sql prompt|>\n",
    """You are an expert data analyst adept at writing SQL queries to answer user questions.

You have access to the following relational tables, with schemas below.

Alongside each text column, in parentheses "()" you will see the top 3 values followed by "..." for columns with more than 5 distinct values, or the top 5 values otherwise. For timestamp and date columns you will see the min/max range of the column.

{table_info}

Write a SQL query to answer the following question, first outputting the chain-of-thought reasoning neccessary to go from question to answer, preceded by a "** Chain-of-Thought Reasoning **" header, and then the SQL query itself, preceded by a "** SQL Query **" header.
{input}""",
    "\n<|cot sql answer|>\n** Chain-of-Thought Reasoning **\n\n",
)

DEFAULT_NL_TO_SQL_COT_ERROR_PROMPT = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n",
    DEFAULT_NL_TO_SQL_ERROR_PROMPT[1],
    "\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
)
