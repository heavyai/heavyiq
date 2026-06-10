# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This module contains prompts for generating Natural Language questions

TABLES_TO_NL_QUESTIONS_TEMPLATE = """
You are an expert data analyst adept at asking compelling questions of your data.

You have access to the following relational tables, with schemas below.

Alongside each text column, in parentheses "()" you will see the top 3 values followed by "..." for columns with more than 5 distinct values, or the top 5 values otherwise. For timestamp and date columns you will see the min/max range of the column.

{table_info}

Write a compelling question to ask of the above data:

Question:"""


TABLES_TO_NL_QUESTIONS_CUSTOM_TEMPLATE = """{prompt_token}
You are an expert data analyst adept at asking compelling questions of your data.

You have access to the following relational tables, with schemas below.

Alongside each text column, in parentheses "()" you will see the top 3 values followed by "..." for columns with more than 5 distinct values, or the top 5 values otherwise. For timestamp and date columns you will see the min/max range of the column.

{table_info}

Write a compelling question to ask of the above data:
{answer_token}
"""
