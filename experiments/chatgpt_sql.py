# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import heavyai
import openai
import re
import sys


def get_table_schema(con, table_name):
    res = con.execute(f"SHOW CREATE TABLE {table_name}")
    table_schema = list(res)[0][0]
    table_schema = re.sub(r" ENCODING .*\)([,\)])", r"\1", table_schema)
    table_schema = re.sub(r",\n.*SHARED DICTIONARY.*REFERENCES.*\([A-Za-z0-9_]*\)", "", table_schema)
    table_schema = re.sub(r"\n", "", table_schema)
    return table_schema


def get_table_str_cols(con, table_name):
    table_details = con.get_table_details(table_name)
    return [col.name for col in table_details if col.type == "STR" and col.encoding == "DICT"]


def get_top_k_vals(con, table_name, column_name, top_k):
    sql = f"""SELECT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL GROUP BY {column_name} ORDER BY COUNT(*) DESC LIMIT {top_k}"""
    results = list(con.execute(sql))
    return [r[0] for r in results]


def get_top_k_vals_str(top_k_vals):
    top_k_vals_strs = [
        f"The most common {len(values)} values for column {key} are: {', '.join(values)}."
        for key, values in top_k_vals.items()
    ]
    return "\n".join(top_k_vals_strs)


def main(argv):
    openai.api_key = "<INSERT_API_KEY>"
    con = heavyai.connect(user="admin", password="HyperInteractive", host="100.85.68.75", dbname="heavyai")
    tables = con.get_tables()
    print("HEAVY.AI Tables")
    table_set = {table.lower() for table in tables}
    print("\n")

    table_input = ""
    max_sql_try_count = 3
    top_k = 20

    top_k_vals_str_cache = {}

    while table_input != "quit":
        table_input = input("Table: ").lower()
        if table_input == "quit":
            break
        if table_input not in table_set:
            print(f"Did not find table {table_input}")
            continue

        table_schema = get_table_schema(con, table_input)
        top_k_str_col_vals_str = top_k_vals_str_cache.get(table_input)
        if top_k_str_col_vals_str is None:
            table_str_cols = get_table_str_cols(con, table_input)
            str_cols_top_k_vals = {
                str_col: get_top_k_vals(con, table_input, str_col, top_k) for str_col in table_str_cols
            }
            top_k_str_col_vals_str = get_top_k_vals_str(str_cols_top_k_vals)
            top_k_vals_str_cache[table_input] = top_k_str_col_vals_str
        print(top_k_str_col_vals_str)

        user_input = input("<<< ")

        prompt = f"""A SQL table named {table_input} is stored in the database with the following schema:\n\n {table_schema}\n\n{top_k_str_col_vals_str}\n\n Translate the following instruction into a SQL query on this table. Exclude null values from the results. Only output a SQL query and nothing else."""
        print(prompt)

        sql_try_count = 1
        sql_success = False
        modified_prompt = copy.deepcopy(prompt)
        while sql_try_count <= max_sql_try_count and not sql_success:
            sql_completion = None
            try:
                sql_completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "system", "content": modified_prompt}, {"role": "user", "content": user_input}],
                )
                sql = sql_completion["choices"][0]["message"]["content"].replace("\n", " ").replace("\r", " ")
                sql = re.sub(".*(SELECT.*;).*", r"\1", sql, count=0, flags=0)
                print("""SQL {sql_try_count}: {sql}""".format(sql_try_count=sql_try_count, sql=sql))
                sql_result = con.execute(sql)
                sql_try_count += 1
                sql_success = True
            except Exception as e:
                print(e)
                modified_prompt += "\nThe last generated SQL statement: {sql} received the following error from the database: {error}.\nPlease rewrite the SQL to avoid this error."
                sql_try_count += 1

            if not sql_success:
                continue

            column_names = [d[0] for d in sql_result.description]
            answer_prompt = """From the above SQL query {sql} in response to the following question: "{user_input}", we received the following result back from the database. Explain this result in plain English. Give a brief answer to the question without explanation of the SQL.""".format(
                sql=sql, user_input=user_input
            )
            long_answer = ""

            results = list(sql_result)
            print(results)
            for row_idx, row in enumerate(results):
                for col_idx, c in enumerate(column_names):
                    long_answer += """Row {row_idx} of {column_name} is {result}. """.format(
                        row_idx=row_idx + 1, column_name=c, result=results[row_idx][col_idx]
                    )
            print(long_answer)
            english_answer = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": answer_prompt}, {"role": "user", "content": long_answer}],
            )
            print(english_answer["choices"][0]["message"]["content"])
            print("\n")


if __name__ == "__main__":
    main(sys.argv[1:])
