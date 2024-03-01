# This module contains prompts for generating Natural Language questions

TABLES_TO_NL_QUESTIONS_TEMPLATE = """
You are an experienced data analyst adept at asking compelling questions of your data.
You have access to the following relational tables, with schemas below.

{table_info}

Write a compelling question to ask of the above data:

Question:"""


TABLES_TO_NL_QUESTIONS_CUSTOM_TEMPLATE = """<|table question prompt|>
You are an experienced data analyst adept at asking compelling questions of your data.
You have access to the following relational tables, with schemas below.

{table_info}

Write a compelling question to ask of the above data:
<|table question prompt|>
"""
