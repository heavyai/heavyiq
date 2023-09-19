import argparse
import copy
import csv
import heavyai
import json
import openai
import pandas as pd
import re
import sqlparse
import sys
from transformers import LlamaTokenizer
from typing import Dict, List, Tuple

from collections.abc import Iterable


def getOptions(argv=None):
    parser = argparse.ArgumentParser(description="Test Spider SQL queries against HeavyDB")
    parser.add_argument("-s", "--host", help="HeavyDB server address", default="localhost")
    parser.add_argument("-p", "--port", help="HeavyDB server port", default="6273")
    parser.add_argument("-u", "--user", help="HeavyDB user name", default="admin")
    parser.add_argument("-w", "--password", help="HeavyDB password", default="HyperInteractive")
    parser.add_argument("-q", "--queries", help="Queries CSV file", default=None)
    parser.add_argument("-e", "--errors", help="Error Queries CSV file", default=None)
    parser.add_argument("-d", "--database", help="HeavyDB database", default=None)
    parser.add_argument("-k", "--openai-api-key", help="OpenAI API Key", default=None)
    parser.add_argument("-c", "--cache", help="Query cache file", default=None)
    parser.add_argument("--start-db", help="Start from this database idx", type=int, default=None)
    parser.add_argument("--num-dbs", help="Only process this many databases", type=int, default=0)
    parser.add_argument(
        "--fix-queries", help="Try to fix failed queries with automatically applied corrections", action="store_true"
    )
    parser.add_argument("--filter-null-groups", help="Add filters to remove null groups", action="store_true")
    parser.add_argument("--fix-queries-gpt", help="Try to fix failed queries with ChatGPT API", action="store_true")
    parser.add_argument("--low-card-top-k-str-vals", help="Low cardinality top K string values", type=int, default=5)
    parser.add_argument("--high-card-top-k-str-vals", help="High cardinality top K string values", type=int, default=3)
    parser.add_argument("--top-k-str-vals", help="Top K string values", type=int, default=5)
    parser.add_argument("--max-instruction-tokens", help="Max instruction tokens", type=int, default=704)
    parser.add_argument(
        "--write-sql-prompts", help="Write SQL prompts to file for successful queries", action="store_true"
    )
    parser.add_argument(
        "--write-english-prompts", help="Write English prompts to file for successful queries", action="store_true"
    )
    parser.add_argument(
        "--write-question-prompts", help="Write question prompts to file for successful queries", action="store_true"
    )
    parser.add_argument(
        "--add-columns-to-question-prompts", help="Add column specifications to question prompts", action="store_true"
    )
    parser.add_argument("--add-columns-to-answers", help="Add column specifications to answer", action="store_true")
    return parser.parse_args(argv)


# List of SQL keywords to be capitalized
sql_keywords = [
    "SELECT",
    "AS",
    "FROM",
    "JOIN",
    "ON",
    "INNER",
    "LEFT",
    "RIGHT",
    "OUTER",
    "WHERE",
    "AND",
    "OR",
    "IN",
    "NOT",
    "NULL",
    "IS",
    "LIKE",
    "EXISTS",
    "ALL",
    "ANY",
    "DISTINCT",
    "GROUP BY",
    "ORDER BY",
    "ASC",
    "DESC",
    "LIMIT",
    "AVG",
    "SUM",
    "COUNT",
    "MAX",
    "MIN",
    "ROUND",
    "LENGTH",
    "LOWER",
    "UPPER",
    "COALESCE",
    "IFNULL",
    "NULLIF",
    "CAST",
    "CONVERT",
    "CASE",
    "WHEN",
    "THEN",
    "ELSE",
    "END",
    "BETWEEN",
]


def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


def parse_keyword(token):
    parsed_tokens = []
    if token.is_group:
        # Recursively handle keywords in the group
        for inner in token.tokens:
            parsed_tokens.append(parse_keyword(inner))
    else:
        if token.value.upper() in sql_keywords:
            parsed_tokens.append(token.value.upper())
        else:
            parsed_tokens.append(token.value)
    return parsed_tokens


def uppercase_sql_keywords(sql):
    # Parse the SQL
    parsed = sqlparse.parse(sql)
    # For each token in the parsed SQL
    parsed_tokens = []
    for stmt in parsed:
        for token in stmt.tokens:
            parsed_tokens.append(parse_keyword(token))

    parsed_tokens = flatten(parsed_tokens)
    new_sql = "".join(token for token in parsed_tokens)
    return new_sql


def normalize_order_by(sql):
    pattern = re.compile(
        r"ORDER BY (?:\w+\([^\)]*\)|\w+\.\w+|\w+)(?: (?:ASC|DESC) NULLS LAST)?(?:, (?:\w+\([^\)]*\)|\w+\.\w+|\w+)(?: (?:ASC|DESC) NULLS LAST)?)*"
    )
    if "ORDER BY" in sql and "ASC" not in sql and "DESC" not in sql:
        for p_str in pattern.findall(sql):
            sql = sql.replace(p_str, p_str + " ASC")
    # Technically we could have a keyword start with ASC OR DESC,
    # but that doesn't happen on the augmented Spider dataset so
    # we don't handle that case for now
    if " ASC" in sql and "ASC NULLS LAST" not in sql:
        sql = sql.replace(" ASC", " ASC NULLS LAST")
    if " DESC" in sql and "DESC NULLS LAST" not in sql:
        sql = sql.replace(" DESC", " DESC NULLS LAST")

    return sql


def adjust_identifier_case(table_statements: list[str], query: str) -> str:
    # Build a map for all column names
    column_map = {}
    table_map = {}
    for statement in table_statements:
        parsed = sqlparse.parse(statement)[0]

        # Extract and add the table name to the table map
        table_name = str(parsed.tokens[4]).split("(")[0]
        table_map[table_name.lower()] = table_name

        # Extract the column definitions from the statement using a regex
        # The regex matches any string that does not contain parentheses
        columns_and_definitions = re.findall(r"\((.*?)\)\s*;", statement)[-1]
        columns = columns_and_definitions.split(",")

        # Add the columns to the column map
        for column in columns:
            column_name = column.split()[0]
            column_map[column_name.lower()] = column_name

    # used_tables = set()
    # used_tables_list = []

    # Split the query and replace column names with their original case
    select_parts = query.split()
    for i, part in enumerate(select_parts):
        part_stripped = part.rstrip(",;")
        # Only apply transformations if the part is not a string literal
        if not (part_stripped.startswith("'") and part_stripped.endswith("'")):
            # Strip trailing commas if they exist
            # Check if the part is a function call
            if "(" in part_stripped and ")" in part_stripped:
                function_name, rest = part_stripped.split("(", 1)
                column_name, rest = rest.rsplit(")", 1)
                if column_name.lower() in column_map:
                    select_parts[i] = (
                        function_name
                        + "("
                        + column_map[column_name.lower()]
                        + ")"
                        + rest
                        + ("," if part[-1] == "," else "")
                        + (";" if part[-1] == ";" else "")
                    )
            elif "." in part_stripped:
                table, column = part_stripped.split(".")
                if column.lower() in column_map:
                    # Preserve the trailing comma if it was present
                    select_parts[i] = f"{table}.{column_map[column.lower()]}" + ("," if part[-1] == "," else "")
            elif part_stripped.lower() in column_map:
                # Preserve the trailing comma if it was present
                select_parts[i] = column_map[part_stripped.lower()] + ("," if part[-1] == "," else "")
            elif part_stripped.lower() in table_map:
                select_parts[i] = table_map[part_stripped.lower()] + (";" if part[-1] == ";" else "")
                # if part_stripped.lower() not in used_tables:
                #    used_tables_list.append(table_map[part_stripped.lower()])
                #    used_tables.add(table_map[part_stripped.lower()])

    # Combine the parts again
    output_query = " ".join(select_parts)
    return output_query
    # return {"query": output_query, "tables": used_tables_list}


def adjust_alias_case(query: str) -> str:
    max_num_aliases = 7
    for alias_idx in range(1, max_num_aliases + 1):
        table_alias1 = f"AS t{alias_idx}"
        table_alias2 = f" t{alias_idx}."
        table_alias3 = f"(t{alias_idx}."
        upper_table_alias_1 = table_alias1.upper()
        upper_table_alias_2 = table_alias2.upper()
        upper_table_alias_3 = table_alias3.upper()
        query = query.replace(table_alias1, upper_table_alias_1)
        query = query.replace(table_alias2, upper_table_alias_2)
        query = query.replace(table_alias3, upper_table_alias_3)
    return query


def add_as_before_table_aliases(query: str) -> str:
    max_num_aliases = 7
    for alias_idx in range(1, max_num_aliases + 1):
        table_alias = f" T{alias_idx} "
        as_table_alias = f" AS T{alias_idx} "
        query = query.replace(table_alias, as_table_alias)
        double_as_table_alias = f" AS AS T{alias_idx} "
        query = query.replace(double_as_table_alias, as_table_alias)
    return query


def add_spaces_around_parentheses(sql_query):
    # Add space before parentheses if not present
    sql_query = re.sub(r"(?<=[^\s])([\(\)])", r" \1", sql_query)

    # Add space after parentheses if not present
    sql_query = re.sub(r"([\(\)])(?=[^\s])", r"\1 ", sql_query)

    # Remove extra spaces before and after parentheses
    sql_query = re.sub(r"\s+([\(\)])\s+", r" \1 ", sql_query)

    return sql_query


def remove_spaces_around_parentheses(sql_query):
    sql_query = sql_query.replace("( ", "(")
    sql_query = sql_query.replace(" )", ")")
    sql_query = sql_query.replace("COUNT (", "COUNT(")
    sql_query = sql_query.replace("SUM (", "SUM(")
    sql_query = sql_query.replace("AVG (", "AVG(")
    sql_query = sql_query.replace("MIN (", "MIN(")
    sql_query = sql_query.replace("MAX (", "MAX(")
    return sql_query


def remove_spaces_around_commas(sql_query):
    sql_query = sql_query.replace(" ,", ",")
    sql_query = sql_query.replace("  ,", ",")
    sql_query = sql_query.replace(",   ", ", ")
    sql_query = sql_query.replace(",  ", ", ")
    return sql_query


def remove_join_aliases(table_statements: list[str], query: str) -> str:
    column_map = {}
    table_map = {}
    for statement in table_statements:
        parsed = sqlparse.parse(statement)[0]

        # Extract and add the table name to the table map
        table_name = str(parsed.tokens[4]).split("(")[0]
        table_map[table_name.lower()] = table_name

        # Extract the column definitions from the statement using a regex
        # The regex matches any string that does not contain parentheses
        columns_and_definitions = re.findall(r"\((.*?)\)\s*;", statement)[-1]
        columns = columns_and_definitions.split(",")

        # Add the columns to the column map
        for column in columns:
            column_name = column.split()[0]
            column_map[column_name.lower()] = column_name
    lower_query = query.lower()
    table_alias_map = {}
    for lower_table_name, table_name in table_map.items():
        alias_prefix = " " + lower_table_name + " as "
        alias_prefix_idx = lower_query.find(alias_prefix)
        if alias_prefix_idx >= 0:
            alias_suffix_idx = alias_prefix_idx + len(alias_prefix)
            alias_name = ""
            while alias_suffix_idx < len(query) and query[alias_suffix_idx].isalnum() or query[alias_suffix_idx] == "_":
                alias_name += query[alias_suffix_idx]
                alias_suffix_idx += 1
            query = query[: alias_prefix_idx + len(table_name) + 1] + query[alias_suffix_idx:]
            lower_query = query.lower()
            table_alias_map[alias_name] = table_name

    for alias, table_name in table_alias_map.items():
        query = query.replace(alias, table_name)
    return query


def getQueriesByDB(queries_file):
    queries_df = pd.read_csv(queries_file)
    queries_df["modified_sql_query"] = queries_df["modified_sql_query"].astype(str)
    queries_df["english_explanation"] = queries_df["english_explanation"].astype(str)
    queries_by_db = {}
    for index, row in queries_df.iterrows():
        query_id = row["query_id"]
        db_id = row["db_id"]
        original_sql_query = row["original_sql_query"].strip()
        modified_sql_query = row["modified_sql_query"].strip()
        english_explanation = row["english_explanation"].strip()
        question = row["question"]
        data_split = row["dataset"]
        if db_id not in queries_by_db:
            queries_by_db[db_id] = [
                {
                    "query_id": query_id,
                    "data_split": data_split,
                    "question": question,
                    "original_sql_query": original_sql_query,
                    "modified_sql_query": modified_sql_query,
                    "english_explanation": english_explanation,
                }
            ]
        else:
            queries_by_db[db_id].append(
                {
                    "query_id": query_id,
                    "data_split": data_split,
                    "question": question,
                    "original_sql_query": original_sql_query,
                    "modified_sql_query": modified_sql_query,
                    "english_explanation": english_explanation,
                }
            )
    return queries_by_db


def get_table_schema(con, table_name):
    res = con.execute(f"SHOW CREATE TABLE {table_name}")
    table_schema = list(res)[0][0]
    table_schema = re.sub(r" ENCODING .*\)([,\)])", r"\1", table_schema)
    table_schema = re.sub(r",\n.*SHARED DICTIONARY.*REFERENCES.*\([A-Za-z0-9_]*\)", "", table_schema)
    table_schema = re.sub(r"\n", "", table_schema)
    table_schema = re.sub(r"\s+", " ", table_schema)
    table_schema = re.sub(r"\(\s+", "(", table_schema)
    table_schema = re.sub(r"\s*WITH \(.*\);?", ";", table_schema)

    return table_schema


def get_table_cols(con, table_name):
    table_details = con.get_table_details(table_name)
    cols = []
    for col in table_details:
        cols.append(table_name + "." + col.name)
    return cols


def get_table_str_cols(con, table_name):
    table_details = con.get_table_details(table_name)
    return [col.name for col in table_details if col.type == "STR" and col.encoding == "DICT" and not col.is_array]


def get_top_k_vals(con, table_name, column_name, top_k):
    sql = f"""SELECT "{column_name}" FROM "{table_name}" WHERE "{column_name}" IS NOT NULL GROUP BY "{column_name}" ORDER BY COUNT(*) DESC LIMIT {top_k}"""
    results = list(con.execute(sql))
    return [r[0] for r in results]


def get_high_low_card_top_k_vals(con, table_name, column_name, low_card_top_k, high_card_top_k):
    count_distinct_sql = (
        f"""SELECT COUNT(DISTINCT "{column_name}") FROM "{table_name}" WHERE "{column_name}" IS NOT NULL"""
    )
    cardinality = list(con.execute(count_distinct_sql))[0][0]
    top_k = high_card_top_k
    if cardinality <= low_card_top_k:
        top_k = low_card_top_k

    top_k_vals_sql = f"""SELECT "{column_name}" FROM "{table_name}" WHERE "{column_name}" IS NOT NULL GROUP BY "{column_name}" ORDER BY COUNT(*) DESC LIMIT {top_k}"""
    top_k_result = list(con.execute(top_k_vals_sql))
    top_k_vals = [r[0] for r in top_k_result]
    return {"table_name": table_name, "column_name": column_name, "cardinality": cardinality, "top_k_vals": top_k_vals}


def get_top_k_vals_str(table_name, top_k_vals):
    print(top_k_vals)
    # top_k_vals_strs = [
    #    f"The most common {len(values)} values for column {key} are: {', '.join(values)}."
    #    for key, values in top_k_vals.items()
    # ]
    top_k_vals_strs = [f"{table_name}.{key}: {', '.join(values['top_k_vals'])}" for key, values in top_k_vals.items()]
    return top_k_vals_strs
    # return "\n".join(top_k_vals_strs)


def generate_instruction(table_schemas, top_k_str_vals, user_question=None):
    # print(top_k_str_vals)
    top_k_str_vals_str = ""
    if top_k_str_vals is not None and len(top_k_str_vals) > 0:
        top_k_str_vals_str = "Sample values for TEXT columns (comma-separated):\n"
        for table_top_k_str_vals in top_k_str_vals:
            top_k_str_vals_str += "\n".join(table_top_k_str_vals)
            top_k_str_vals_str += "\n"
    if user_question is not None:
        instruction = """You are a experienced data analyst adept at writing SQL queries to answer user questions.\nYou have access to the following relational tables, with schemas below.\n{table_schemas}\n\n{top_k_str_vals_str}\nWrite a SQL query to answer the following question:\n\n{user_question}\n""".format(
            table_schemas="\n\n".join(table_schemas), top_k_str_vals_str=top_k_str_vals_str, user_question=user_question
        )
        return instruction
    else:
        instruction = """You are a experienced data analyst adept at asking compelling questions of your data.\nYou have access to the following relational tables, with schemas below.\n\n{table_schemas}\n\n{top_k_str_vals_str}\nWrite a compelling question to ask of the above data:\n""".format(
            table_schemas="\n\n".join(table_schemas), top_k_str_vals_str=top_k_str_vals_str
        )
        return instruction


def generate_question(table_schemas, top_k_str_vals, unique_columns=None):
    top_k_str_vals_str = ""
    if top_k_str_vals is not None and len(top_k_str_vals) > 0:
        top_k_str_vals_str = "Sample values for TEXT columns (comma-separated):\n"
        for table_top_k_str_vals in top_k_str_vals:
            top_k_str_vals_str += "\n".join(table_top_k_str_vals)
            top_k_str_vals_str += "\n"
    if unique_columns is not None:
        unique_columns_str = ""
        if len(unique_columns) > 0:
            for unique_column in unique_columns:
                unique_columns_str += unique_column[1] + "." + unique_column[2] + "\n"
        else:
            unique_columns_str += "COUNT(*)\n"
        instruction = """You are a experienced data analyst adept at asking compelling questions of your data.\nYou have access to the following relational tables, with schemas below.\n\n{table_schemas}\n\n{top_k_str_vals_str}\nUse the following columns to generate the question a compelling question of the data:\n\n{unique_columns_str}\n""".format(
            table_schemas="\n\n".join(table_schemas),
            top_k_str_vals_str=top_k_str_vals_str,
            unique_columns_str=unique_columns_str,
        )
        return instruction
    else:
        instruction = """You are a experienced data analyst adept at asking compelling questions of your data.\nYou have access to the following relational tables, with schemas below.\n\n{table_schemas}\n\n{top_k_str_vals_str}\n\nWrite a compelling question to ask of the above data:\n""".format(
            table_schemas="\n\n".join(table_schemas), top_k_str_vals_str=top_k_str_vals_str
        )
        return instruction


def extract_tables_from_query(con, query):
    explain_query = "EXPLAIN CALCITE " + query
    query_plan = con.execute(explain_query)
    query_plan = list(query_plan)[0][0]
    pattern = r"LogicalTableScan\(table=\[\[(.*?)\]\]\)"
    matches = re.findall(pattern, query_plan)
    tables = []
    for match in matches:
        table = match.split(", ")[1]
        tables.append(table)
    return list(set(tables))  # remove duplicates


def write_sql_prompts_to_csv(prompts, output_file):
    fieldnames = [
        "query_id",
        "db_id",
        "tables",
        "data_split",
        "instruction_tokens",
        "targeted_instruction_tokens",
        "output_tokens",
        "instruction",
        "targeted_instruction",
        "output",
    ]
    prompts_to_write = [
        (
            obj["query_id"],
            obj["db_id"],
            obj["tables"],
            obj["data_split"],
            obj["instruction_tokens"],
            obj["targeted_instruction_tokens"],
            obj["output_tokens"],
            obj["instruction"],
            obj["targeted_instruction"],
            obj["output"],
        )
        for obj in prompts
    ]
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(fieldnames)
        writer.writerows(prompts_to_write)


def write_english_prompts_to_csv(english_prompts, output_file):
    fieldnames = [
        "query_id",
        "db_id",
        "data_split",
        "instruction",
        "output",
    ]
    prompts_to_write = [
        (
            obj["query_id"],
            obj["db_id"],
            obj["data_split"],
            obj["instruction"],
            obj["output"],
        )
        for obj in english_prompts
    ]
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(fieldnames)
        writer.writerows(prompts_to_write)


def write_question_prompts_to_csv(question_prompts, output_file):
    fieldnames = [
        "query_id",
        "db_id",
        "data_split",
        "instruction",
        "output",
    ]
    prompts_to_write = [
        (
            obj["query_id"],
            obj["db_id"],
            obj["data_split"],
            obj["instruction"],
            obj["output"],
        )
        for obj in question_prompts
    ]
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(fieldnames)
        writer.writerows(prompts_to_write)


def extract_group_by_expressions(query):
    # query = query.upper()
    if "GROUP BY" not in query:
        return []

    # This pattern assumes that no other part of the query contains 'GROUP BY'
    pattern = r"GROUP BY(.+?)( ORDER BY| HAVING|;|$)"
    group_by_clause = re.search(pattern, query)

    if group_by_clause:
        # Separate multiple group by expressions by comma and strip whitespace
        expressions = group_by_clause.group(1).split(",")
        expressions = [expr.strip() for expr in expressions]
        return expressions

    return []


def extract_projected_expressions(query):
    # query = query.upper()
    pattern = r"SELECT(.+?)FROM"
    select_clause = re.search(pattern, query)

    if select_clause:
        # Separate multiple projected expressions by comma and strip whitespace
        expressions = select_clause.group(1).split(",")
        expressions = [expr.strip() for expr in expressions]
        return expressions

    return []


def add_to_select_projection(query, new_items):
    # query = query.upper()
    pattern = r"SELECT(.+?)FROM"
    select_clause = re.search(pattern, query)

    if select_clause:
        # Separate multiple projected expressions by comma and strip whitespace
        expressions = select_clause.group(1).split(",")
        expressions = [expr.strip() for expr in expressions]

        # Add new items
        expressions.extend(new_items)

        # Reconstruct the SELECT clause
        new_select_clause = ", ".join(expressions)

        # Replace the original SELECT clause with the new one
        query = re.sub(pattern, f"SELECT {new_select_clause} FROM", query)

    return query


def add_to_group_by(query, new_items):
    # query = query.upper()
    pattern = r"GROUP BY(.+?)( ORDER BY| HAVING|;|$)"
    group_by_clause = re.search(pattern, query)

    if group_by_clause:
        # Separate multiple group by expressions by comma and strip whitespace
        expressions = group_by_clause.group(1).split(",")
        expressions = [expr.strip() for expr in expressions]

        # Add new items
        expressions.extend(new_items)

        # Reconstruct the GROUP BY clause
        new_group_by_clause = ", ".join(expressions)

        # Replace the original GROUP BY clause with the new one
        query = re.sub(pattern, f"GROUP BY {new_group_by_clause}\\2", query)

    return query


def fix_failed_query_unquoted_keyword(con, query_id, failed_query, error):
    reserved_keywords = [
        "YEAR",
        "year",
        "Year" "DATE",
        "date",
        "Date" "MONTH",
        "month",
        "Month" "DAY",
        "day",
        "Day" "RANK",
        "rank",
        "Rank" "LENGTH",
        "length",
        "Length",
    ]
    sql_query = copy.deepcopy(failed_query)
    for reserved_keyword in reserved_keywords:
        if reserved_keyword in sql_query:
            sql_query = sql_query.replace(reserved_keyword, f'"{reserved_keyword}"')
    try:
        sql_result = con.execute(sql_query)
        return sql_query
    except Exception as error:
        return None


def fix_failed_query_implicit_group_by(con, query_id, failed_query, error):
    sql_query = copy.deepcopy(failed_query)
    group_by_exprs = extract_group_by_expressions(sql_query)
    if len(group_by_exprs) == 0:
        return None
    projected_exprs = extract_projected_expressions(sql_query)
    if len(projected_exprs) == 0:
        return None
    group_by_exprs_to_add = []
    agg_exprs = ["AVG", "COUNT", "MAX", "MIN", "SUM"]
    for projected_expr in projected_exprs:
        found_agg_expr = False
        for agg_expr in agg_exprs:
            if projected_expr.upper().startswith(agg_expr):
                found_agg_expr = True
                break
        if found_agg_expr:
            continue
        if projected_expr not in group_by_exprs:
            group_by_exprs_to_add.append(projected_expr)
    if len(group_by_exprs_to_add) == 0:
        return None
    sql_query = add_to_group_by(sql_query, group_by_exprs_to_add)
    try:
        sql_result = con.execute(sql_query)
        return sql_query
    except Exception as error:
        return None


def fix_failed_query_gpt(con, query_id, failed_query, error):
    sql = copy.deepcopy(failed_query)
    prompt = f"""When translating the following SQL query to work in the OmniSciDB database, the following error was received.\n{error}\n\n.Note that OmniSciDB has the following limitations:\n1.INTERSECT, EXCEPT, and UNION claues are not supported (UNION ALL is supported, however).\n2.Database reserved keywords like "YEAR", "MONTH", "DAY", "RANK" and "LENGTH" need to be enclosed in double-quotes.\n3.Implicit group-by is not supported, meaning that all projected columns in a group-by query must either be in the GROUP BY clause or be aggregates.\n3. Columns of any numeric type cannot be joined with columns of type TEXt. To join these columns, either the numeric column must be casted to TEXT via a CAST operator, or the TEXT column casted to a numeric type with TRY_CAST.\n\nWith these limitations in mind, please alter the following query to run in OmniSciDB: """

    max_sql_try_count = 2
    sql_try_count = 1
    sql_success = False
    modified_prompt = copy.deepcopy(prompt)
    base_model = "gpt-3.5-turbo"
    power_model = "gpt-4"
    temperature = 0.05
    while sql_try_count <= max_sql_try_count and not sql_success:
        sql_completion = None
        model = base_model if sql_try_count == 1 else power_model
        try:
            sql_completion = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "system", "content": modified_prompt}, {"role": "user", "content": sql}],
                temperature=temperature,
            )
            sql = sql_completion["choices"][0]["message"]["content"].replace("\n", " ").replace("\r", " ")
            sql = re.sub(".*(SELECT.*;).*", r"\1", sql, count=0, flags=0)
            # print("""SQL {sql_try_count}: {sql}""".format(sql_try_count=sql_try_count, sql=sql))
            sql_result = con.execute(sql)
            sql_try_count += 1
            sql_success = True
        except Exception as error:
            # print(error)
            prompt = f"""When translating the following SQL query to work in the OmniSciDB database, the following error was received.\n{error}\n\n.Note that OmniSciDB has the following limitations:\n1.INTERSECT, EXCEPT, and UNION claues are not supported (UNION ALL is supported, however).\n2.Database reserved keywords like "YEAR", "MONTH", "DAY", "RANK" and "LENGTH" need to be enclosed in double-quotes.\n3.Implicit group-by is not supported, meaning that all projected columns in a group-by query must either be in the GROUP BY clause or be aggregates.\n3. Columns of any numeric type cannot be joined with columns of type TEXt. To join these columns, either the numeric column must be casted to TEXT via a CAST operator, or the TEXT column casted to a numeric type with TRY_CAST.\n\nWith these limitations in mind, please alter the following query to run in OmniSciDB: """
            sql_try_count += 1
            continue
    if not sql_success:
        print(f"Query {query_id} FAILED: {failed_query}")
        return None
    else:
        print(f"Query {query_id} FIXED: {sql}")
        return sql


def add_not_null_filters(sql_query):
    # Check if the query contains a GROUP BY clause
    # Look for SQL keywords following GROUP BY, so the regular expression will stop at them
    group_by_match = re.search(
        r"GROUP BY\s+([\w_,\s]+)(?=\s+ORDER BY|\s+HAVING|\s+LIMIT|\s+OFFSET|\s+;|$)", sql_query, re.IGNORECASE
    )
    if not group_by_match:
        return sql_query

    # Extract columns in GROUP BY clause
    group_by_columns = group_by_match.group(1).split(",")
    group_by_columns = [col.strip() for col in group_by_columns]

    # Check if the query contains a WHERE clause
    where_match = re.search(
        r"WHERE\s+(.+?)(?=\s+GROUP BY|\s+ORDER BY|\s+HAVING|\s+LIMIT|\s+OFFSET|\s+;|$)", sql_query, re.IGNORECASE
    )

    if where_match:
        where_clause = where_match.group(1)
        new_where_clause = where_clause
        for column in group_by_columns:
            # Append filters only if they are not already there
            if f"{column} IS NOT NULL" not in where_clause:
                new_where_clause = f"{new_where_clause} AND {column} IS NOT NULL"
        # Replace old WHERE clause with the new one
        sql_query = sql_query.replace(where_clause, new_where_clause)
    else:
        # Add a WHERE clause if it does not exist
        where_clause = " AND ".join([f"{column} IS NOT NULL" for column in group_by_columns])
        sql_query = re.sub(r"(FROM\s+\w+)", r"\1 WHERE " + where_clause, sql_query, flags=re.IGNORECASE)

    return sql_query


def load_query_cache(cache_file):
    queries = []
    with open(cache_file, "r") as f:
        queries = json.load(f)
    query_cache_by_db = {}
    for query in queries:
        db_id = query["db_id"]
        sql_query = query["output"]
        if db_id not in query_cache_by_db:
            query_cache_by_db[db_id] = set(sql_query)
        else:
            query_cache_by_db[db_id].add(sql_query)
    return query_cache_by_db


def get_query_plan(con, query):
    explain_calcite_query = f"EXPLAIN CALCITE DETAILED {query}"
    res = con.execute(explain_calcite_query)
    query_plan = list(res)[0][0]
    return query_plan


def extract_column_mappings(query_plan: str) -> Dict[str, Tuple[str, str, str]]:
    result = {}
    # Use a regex pattern to capture (database, table, column) and literal values in LogicalFilter
    # pattern = r"\[\$([0-9]+)->(db:[\w]+),tableName:([\w]+),colName:([\w]+)\]\], \=.*'([^']+)')"
    # pattern = r"\[\$([0-9]+)->(db:[\w]+),tableName:([\w]+),colName:([\w]+)\], \=.*'([^']+)'\)"
    pattern = r"\[\$([0-9]+)->db:([\w]+),tableName:([\w]+),colName:([\w]+)\]"

    matches = re.findall(pattern, query_plan)

    # for _, db, table, column, literal in matches:
    for id, db, table, column in matches:
        col_id = (db, table, column)
        result[id] = col_id

    return result


def get_unique_columns(col_mappings: Dict[str, Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    unique_cols = set(col_mappings.values())
    return sorted(list(unique_cols))


def schema_order_columns(table_cols: List[str], used_cols: List[Tuple[str, str, str]]) -> List[str]:
    used_col_strs = [used_col[1] + "." + used_col[2] for used_col in used_cols]
    sorted_used_cols = sorted(used_col_strs, key=table_cols.index)
    return sorted_used_cols


def generate_error_prompts(error_queries_file, output_file, options):
    errors_df = pd.read_csv(error_queries_file)
    prompts = []
    for index, row in errors_df.iterrows():
        db = row["db_id"]
        con = heavyai.connect(user=options.user, password=options.password, host=options.host, dbname=db)
        db_tables = con.get_tables()
        table_schemas = []
        table_schemas_map = {}
        top_k_str_vals = []
        top_k_str_vals_map = {}
        for db_table in db_tables:
            table_schema = get_table_schema(con, db_table)
            table_schemas.append(table_schema)
            table_schemas_map[db_table.lower()] = table_schema
            if options.top_k_str_vals > 0:
                table_str_cols = get_table_str_cols(con, db_table)
                str_cols_top_k_vals = {
                    str_col: get_high_low_card_top_k_vals(
                        con, db_table, str_col, options.low_card_top_k_str_vals, options.high_card_top_k_str_vals
                    )
                    for str_col in table_str_cols
                }
                top_k_str_col_vals_str = get_top_k_vals_str(db_table, str_cols_top_k_vals)
                top_k_str_vals_map[db_table] = top_k_str_col_vals_str
                top_k_str_vals.append(top_k_str_col_vals_str)
        print(table_schemas)
        print(top_k_str_vals)
        question = row["question"]
        data_split = row["dataset"]
        error_sql_query = row["error_sql_query"]
        error = row["db_error_msg"]
        correct_sql_query = row["correct_sql_query"]

        filtered_tables = extract_tables_from_query(con, correct_sql_query)
        filtered_table_schemas = []
        for db_table in filtered_tables:
            filtered_table_schemas.append(table_schemas_map[db_table.lower()])
        filtered_top_k_str_vals = []
        if options.top_k_str_vals > 0:
            for db_table in filtered_tables:
                filtered_top_k_str_vals.append(top_k_str_vals_map[db_table])
        top_k_str_vals_str = ""
        if filtered_top_k_str_vals is not None and len(filtered_top_k_str_vals) > 0:
            top_k_str_vals_str = "Sample values for TEXT columns (comma-separated):\n"
            for table_top_k_str_vals in filtered_top_k_str_vals:
                top_k_str_vals_str += "\n".join(table_top_k_str_vals)
                top_k_str_vals_str += "\n"

        targeted_instruction = """You generated a SQL query that generated an exception when executed in the HeavyDB database.\n\nYou have access to the following relation tables, with schemas below.\n{table_schemas}\n\n{top_k_str_vals_str}\nIn attempting to answer the following user question:\n\n{question},\nyou generated the following SQL query:\n{error_sql_query}\n,which failed to run in the HeavyDB database, generating the following error:\n{error}\n\nPlease alter the query to run without error in HeavyDB:""".format(
            table_schemas="\n\n".join(filtered_table_schemas),
            top_k_str_vals_str=top_k_str_vals_str,
            question=question,
            error_sql_query=error_sql_query,
            error=error,
        )
        prompts.append(
            {
                "query_id": row["query_id"],
                "db_id": db,
                "tables": filtered_tables,
                "data_split": data_split,
                "instruction": targeted_instruction,
                "output": correct_sql_query,
            }
        )
    fieldnames = [
        "query_id",
        "db_id",
        "tables",
        "data_split",
        "instruction",
        "output",
    ]
    prompts_to_write = [
        (
            obj["query_id"],
            obj["db_id"],
            obj["tables"],
            obj["data_split"],
            obj["instruction"],
            obj["output"],
        )
        for obj in prompts
    ]
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(fieldnames)
        writer.writerows(prompts_to_write)


def main(argv):
    options = getOptions(argv)
    openai.api_key = options.openai_api_key
    if options.errors is not None:
        generate_error_prompts(options.errors, "sql_error_prompts.csv", options)

    if options.queries is not None:
        queries_by_db = getQueriesByDB(options.queries)
        query_cache_by_db = {}
        if options.cache is not None:
            query_cache_by_db = load_query_cache(options.cache)
        db_num = 0
        total_queries = 0
        total_successful_queries = 0
        total_successful_altered_queries = 0
        fixed_queries = []
        failed_queries = []
        prompts = []
        english_prompts = []
        question_prompts = []
        question_prompts_with_cols = []
        con = None
        tokenizer = LlamaTokenizer.from_pretrained("test_model")
        num_null_rewrite_successes = 0
        num_null_rewrite_fails = 0
        null_rewrite_fail_ids = []
        for db_id, queries in queries_by_db.items():
            try:
                if db_num < options.start_db:
                    db_num += 1
                    continue
                try:
                    con = heavyai.connect(user=options.user, password=options.password, host=options.host, dbname=db_id)
                except Exception as e:
                    print(f"Error connecting to database {db_id}: {e}")
                    db_num += 1
                    continue
                db_query_cache = query_cache_by_db[db_id] if db_id in query_cache_by_db else set()
                num_queries = len(queries)
                successful_queries = 0
                db_tables = con.get_tables()
                table_schemas = []
                table_cols = []
                table_schemas_map = {}
                top_k_str_vals_map = {}
                top_k_str_vals = []
                for db_table in db_tables:
                    table_schema = get_table_schema(con, db_table)
                    if options.add_columns_to_answers:
                        table_cols.extend(get_table_cols(con, db_table))
                    table_schemas.append(table_schema)
                    table_schemas_map[db_table.lower()] = table_schema
                    if options.low_card_top_k_str_vals > 0:
                        table_str_cols = get_table_str_cols(con, db_table)
                        str_cols_top_k_vals = {
                            str_col: get_high_low_card_top_k_vals(
                                con,
                                db_table,
                                str_col,
                                options.low_card_top_k_str_vals,
                                options.high_card_top_k_str_vals,
                            )
                            for str_col in table_str_cols
                        }
                        # print(str_cols_top_k_vals)
                        top_k_str_col_vals_str = get_top_k_vals_str(db_table, str_cols_top_k_vals)
                        # print(top_k_str_col_vals_str)
                        top_k_str_vals_map[db_table] = top_k_str_col_vals_str
                        top_k_str_vals.append(top_k_str_col_vals_str)

                for query in queries:
                    query_id = query["query_id"]
                    original_sql_query = query["original_sql_query"]
                    modified_sql_query = query["modified_sql_query"]
                    data_split = query["data_split"]
                    english_explanation = query["english_explanation"]
                    sql_query = modified_sql_query if len(str(modified_sql_query)) > 3 else original_sql_query
                    sql_query = re.sub(" +", " ", sql_query)
                    sql_query = re.sub(" ,", ",", sql_query)
                    sql_query = sql_query + ";" if sql_query[-1] != ";" else sql_query
                    sql_query = uppercase_sql_keywords(sql_query)
                    filtered_tables = None
                    try:
                        sql_query = adjust_identifier_case(table_schemas, sql_query)
                        sql_query = normalize_order_by(sql_query)
                        # sql_query = add_spaces_around_parentheses(sql_query)
                        sql_query = remove_spaces_around_parentheses(sql_query)
                        sql_query = remove_spaces_around_commas(sql_query)
                        sql_query = adjust_alias_case(sql_query)
                        sql_query = add_as_before_table_aliases(sql_query)
                        if options.filter_null_groups:
                            old_sql_query = copy.deepcopy(sql_query)
                            sql_query = add_not_null_filters(sql_query)
                            if sql_query != old_sql_query:
                                try:
                                    con.execute(sql_query)
                                    print(f"Rewrite SUCCESS: {query_id}")
                                    num_null_rewrite_successes += 1
                                except Exception as e:
                                    print(f"Rewrite FAIL: {query_id}")
                                    num_null_rewrite_fails += 1
                                    null_rewrite_fail_ids.append(query_id)
                                    sql_query = old_sql_query
                        # sql_query = remove_join_aliases(table_schemas, sql_query)

                        # query_and_tables = adjust_identifier_case(table_schemas, sql_query)
                        # sql_query = query_and_tables["query"]
                        # filtered_tables = query_and_tables["tables"]
                    except Exception as e:
                        print(f"Exception Query ID: {query_id} DB: {db_id} Query: {sql_query}")
                        print(e)
                        continue
                    # print(f"Query ID: {query_id} DB: {db_id} Query: {sql_query}")
                    sql_query = sql_query + ";" if sql_query[-1] != ";" else sql_query

                    # print(f"Query ID: {query_id}")
                    # print(f"SQL Query: {sql_query}")
                    try:
                        if sql_query not in db_query_cache:
                            con.execute(sql_query)
                        successful_queries += 1
                        if options.write_sql_prompts:
                            filtered_tables = extract_tables_from_query(con, sql_query)
                            # print(f"Query ID: {query_id} Tables: {filtered_tables} Query: {sql_query}")
                            sql_query_tokens = tokenizer.tokenize(sql_query)
                            num_sql_query_tokens = len(sql_query_tokens)
                            filtered_table_schemas = []
                            for db_table in filtered_tables:
                                filtered_table_schemas.append(table_schemas_map[db_table.lower()])
                            filtered_top_k_str_vals = []
                            if options.top_k_str_vals > 0:
                                for db_table in filtered_tables:
                                    filtered_top_k_str_vals.append(top_k_str_vals_map[db_table])
                            instruction = generate_instruction(table_schemas, top_k_str_vals, query["question"])
                            targeted_instruction = generate_instruction(
                                filtered_table_schemas, filtered_top_k_str_vals, query["question"]
                            )
                            instruction_tokens = tokenizer.tokenize(instruction)
                            num_instruction_tokens = len(instruction_tokens)
                            targeted_instruction_tokens = tokenizer.tokenize(targeted_instruction)
                            num_targeted_instruction_tokens = len(targeted_instruction_tokens)
                            if num_targeted_instruction_tokens > options.max_instruction_tokens:
                                targeted_instruction = generate_instruction(
                                    filtered_table_schemas, [], query["question"]
                                )
                                targeted_instruction_tokens = tokenizer.tokenize(targeted_instruction)
                                num_targeted_instruction_tokens = len(targeted_instruction_tokens)
                            if options.add_columns_to_answers:
                                query_plan = get_query_plan(con, sql_query)
                                col_mapping = extract_column_mappings(query_plan)
                                unique_used_columns = get_unique_columns(col_mapping)
                                sorted_used_columns = schema_order_columns(table_cols, unique_used_columns)
                                output = "Columns used in the query:\n"
                                for col in sorted_used_columns:
                                    output += col + "\n"
                                output += "\n\nSQL query:\n" + sql_query
                            else:
                                output = copy.deepcopy(sql_query)

                            sql_query_tokens = tokenizer.tokenize(sql_query)
                            num_sql_query_tokens = len(sql_query_tokens)
                            # sql_query_with_semicolon = sql_query + ";" if sql_query[-1] != ";" else sql_query
                            prompts.append(
                                {
                                    "db_id": db_id,
                                    "query_id": query_id,
                                    "data_split": data_split,
                                    "tables": filtered_tables,
                                    "instruction": instruction,
                                    "targeted_instruction": targeted_instruction,
                                    "output": output,
                                    "instruction_tokens": num_instruction_tokens,
                                    "targeted_instruction_tokens": num_targeted_instruction_tokens,
                                    "output_tokens": num_sql_query_tokens,
                                }
                            )
                        if options.write_english_prompts:
                            if len(english_explanation) > 4:
                                cursor = con.execute(sql_query)
                                results = str(cursor.fetchall())
                                instruction = f"The user asked the following question:\n{query['question']}\n\nTo answer the question, the following SQL query was generated:\n{sql_query}\n\nThe following results were returned:\n\n{results}\n\nNow explain the results in English, referencing the question and the SQL query as needed."
                                english_prompts.append(
                                    {
                                        "db_id": db_id,
                                        "query_id": query_id,
                                        "data_split": data_split,
                                        "instruction": instruction,
                                        "output": english_explanation,
                                    }
                                )
                        if options.write_question_prompts:
                            filtered_tables = extract_tables_from_query(con, sql_query)
                            filtered_table_schemas = []
                            for db_table in filtered_tables:
                                filtered_table_schemas.append(table_schemas_map[db_table.lower()])
                            filtered_top_k_str_vals = []
                            if options.top_k_str_vals > 0:
                                for db_table in filtered_tables:
                                    filtered_top_k_str_vals.append(top_k_str_vals_map[db_table])
                            unique_columns = None
                            if options.add_columns_to_question_prompts:
                                query_plan = get_query_plan(con, sql_query)
                                col_mapping = extract_column_mappings(query_plan)
                                unique_columns = get_unique_columns(col_mapping)
                            instruction = generate_question(
                                filtered_table_schemas, filtered_top_k_str_vals, unique_columns
                            )
                            instruction_tokens = tokenizer.tokenize(instruction)
                            num_instruction_tokens = len(instruction_tokens)
                            if num_instruction_tokens > options.max_instruction_tokens:
                                print(f"Instruction too long: {num_instruction_tokens}")
                                instruction = generate_question(filtered_table_schemas, None, unique_columns)
                            question_prompts.append(
                                {
                                    "db_id": db_id,
                                    "query_id": query_id,
                                    "data_split": data_split,
                                    "instruction": instruction,
                                    "output": query["question"],
                                }
                            )

                    except Exception as e:
                        print(f"Exception Query ID: {query_id} DB: {db_id} Query: {sql_query}")
                        query_fixed = False
                        if options.fix_queries:
                            fixed_query = fix_failed_query_unquoted_keyword(con, query_id, sql_query, e)
                            if fixed_query is None:
                                fixed_query = fix_failed_query_implicit_group_by(con, query_id, sql_query, e)
                            if fixed_query is not None:
                                fixed_queries.append({"query_id": query_id, "db_id": db_id, "sql_query": fixed_query})
                                query_fixed = True
                        if options.fix_queries_gpt and not query_fixed:
                            fixed_query = fix_failed_query_gpt(con, query_id, sql_query, e)
                            if fixed_query is not None:
                                # print(f"Fixed query: {fixed_query}")
                                fixed_queries.append({"query_id": query_id, "db_id": db_id, "sql_query": fixed_query})
                                query_fixed = True
                        if not query_fixed:
                            failed_queries.append({"query_id": query_id, "db_id": db_id, "sql_query": sql_query})
                total_queries += num_queries
                total_successful_queries += successful_queries
                print(f"{db_num}: {db_id} successful queries: {successful_queries}/{num_queries}")
                db_num += 1
                if options.num_dbs != None and db_num >= options.num_dbs + options.start_db:
                    break
            except Exception as e:
                pass
        failures_df = pd.DataFrame(failed_queries)
        failures_df.to_csv("failed_queries.csv", index=False)
        if options.fix_queries or options.fix_queries_gpt:
            fixes_df = pd.DataFrame(fixed_queries)
            fixes_df.to_csv("fixed_queries.csv", index=False)
        if options.write_sql_prompts:
            with open("sql_all_prompts.json", "w") as file:
                json.dump(prompts, file, indent=4)
            write_sql_prompts_to_csv(prompts, "sql_all_prompts.csv")
        if options.write_english_prompts:
            write_english_prompts_to_csv(english_prompts, "sql_english_prompts.csv")
        if options.write_question_prompts:
            write_question_prompts_to_csv(question_prompts, "sql_question_prompts.csv")

        print(f"\n\nTotal NULL Rewrite SUCCESSES: {num_null_rewrite_successes}, FAILS: {num_null_rewrite_fails}")
        print(f"\n\nTotal successful queries: {total_successful_queries}/{total_queries}")
        print(f"Total successful fixed queries: {len(fixed_queries)}/{total_queries}")
        print(", ".join(str(id) for id in null_rewrite_fail_ids))
        # print(f"Total successful altered queries: {total_successful_altered_queries}/{total_queries}")


if __name__ == "__main__":
    main(sys.argv[1:])
