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

from collections.abc import Iterable


def getOptions(argv=None):
    parser = argparse.ArgumentParser(description="Test Spider SQL queries against HeavyDB")
    parser.add_argument("-s", "--host", help="HeavyDB server address", default="localhost")
    parser.add_argument("-p", "--port", help="HeavyDB server port", default="6273")
    parser.add_argument("-u", "--user", help="HeavyDB user name", default="admin")
    parser.add_argument("-w", "--password", help="HeavyDB password", default="HyperInteractive")
    parser.add_argument("-q", "--queries", help="Queries CSV file", default="./spider_qa.csv")
    parser.add_argument("-d", "--database", help="HeavyDB database", default=None)
    parser.add_argument("-k", "--openai-api-key", help="OpenAI API Key", default=None)
    parser.add_argument("--start-db", help="Start from this database idx", type=int, default=None)
    parser.add_argument("--num-dbs", help="Only process this many databases", type=int, default=0)
    parser.add_argument(
        "--fix-queries", help="Try to fix failed queries with automatically applied corrections", action="store_true"
    )
    parser.add_argument("--fix-queries-gpt", help="Try to fix failed queries with ChatGPT API", action="store_true")
    parser.add_argument("--write-prompts", help="Write prompts to file for successful queries", action="store_true")
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

    # Combine the parts again
    output_query = " ".join(select_parts)
    return output_query

def getQueriesByDB(queries_file):
    queries_df = pd.read_csv(queries_file)
    queries_df["modified_sql_query"] = queries_df["modified_sql_query"].astype(str)
    queries_by_db = {}
    for index, row in queries_df.iterrows():
        query_id = row["query_id"]
        db_id = row["db_id"]
        original_sql_query = row["original_sql_query"]
        modified_sql_query = row["modified_sql_query"]
        question = row["question"]
        if db_id not in queries_by_db:
            queries_by_db[db_id] = [
                {
                    "query_id": query_id,
                    "question": question,
                    "original_sql_query": original_sql_query,
                    "modified_sql_query": modified_sql_query,
                }
            ]
        else:
            queries_by_db[db_id].append(
                {
                    "query_id": query_id,
                    "question": question,
                    "original_sql_query": original_sql_query,
                    "modified_sql_query": modified_sql_query,
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

    return table_schema


def generate_instruction(table_schemas, user_question):
    instruction = """You are a experienced data analyst adept at writing SQL queries to answer user questions.\n
You have access to the following relational tables, with schemas below.\n
{table_schemas}\n
Write a SQL query to answer the following question:\n
{user_question}\n""".format(
        table_schemas="\n\n".join(table_schemas), user_question=user_question
    )
    return instruction


def write_prompts_to_csv(prompts, output_file):
    fieldnames = ["query_id", "db_id", "instruction", "output"]
    prompts_to_write = [(obj["query_id"], obj["db_id"], obj["instruction"], obj["output"]) for obj in prompts]
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
        print(sql_try_count)
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


def main(argv):
    options = getOptions(argv)
    openai.api_key = options.openai_api_key
    queries_by_db = getQueriesByDB(options.queries)
    db_num = 0
    total_queries = 0
    total_successful_queries = 0
    total_successful_altered_queries = 0
    fixed_queries = []
    failed_queries = []
    prompts = []
    con = None
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
            num_queries = len(queries)
            successful_queries = 0
            db_tables = con.get_tables()
            table_schemas = []
            for db_table in db_tables:
                table_schemas.append(get_table_schema(con, db_table))
            for query in queries:
                query_id = query["query_id"]
                original_sql_query = query["original_sql_query"]
                modified_sql_query = query["modified_sql_query"]
                sql_query = modified_sql_query if len(str(modified_sql_query)) > 3 else original_sql_query
                #print(query_id)
                #print(sql_query)
                sql_query = re.sub(" +", " ", sql_query)
                sql_query = re.sub(" ,", ",", sql_query)
                sql_query = sql_query + ";" if sql_query[-1] != ";" else sql_query
                sql_query = uppercase_sql_keywords(sql_query)
                #print(sql_query)
                try:
                    sql_query = adjust_identifier_case(table_schemas, sql_query)
                except Exception as e:
                    print(f"Query ID: {query_id} DB: {db_id} Query: {sql_query}")
                    print(e)
                    continue
                #print(f"Query ID: {query_id} DB: {db_id} Query: {sql_query}")

                # print(f"Query ID: {query_id}")
                # print(f"SQL Query: {sql_query}")
                try:
                    results = list(con.execute(sql_query))
                    successful_queries += 1
                    if options.write_prompts:
                        instruction = generate_instruction(table_schemas, query["question"])
                        # sql_query_with_semicolon = sql_query + ";" if sql_query[-1] != ";" else sql_query
                        prompts.append(
                            {"instruction": instruction, "output": sql_query, "db_id": db_id, "query_id": query_id}
                        )
                except Exception as e:
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

                    # if '"' in sql_query:
                    #    sql_query = sql_query.replace('"', "'")
                    #    try:
                    #        results = list(con.execute(sql_query))
                    #        successful_queries += 1
                    #        total_successful_altered_queries += 1
                    #    except Exception as e:
                    #        failed_queries.append({"query_id": query_id, "db_id": db_id, "sql_query": sql_query})
                    #        continue
                    # else:
                    #    # failed_queries.append({"query_id": query_id, "db_id": db_id, "sql_query": sql_query, "error": e})
                    #    failed_queries.append({"query_id": query_id, "db_id": db_id, "sql_query": sql_query})
                    #    continue
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
    if options.write_prompts:
        with open("sql_all_prompts.json", "w") as file:
            json.dump(prompts, file, indent=4)
        write_prompts_to_csv(prompts, "sql_all_prompts.csv")
    print(f"\n\nTotal successful queries: {total_successful_queries}/{total_queries}")
    print(f"Total successful fixed queries: {len(fixed_queries)}/{total_queries}")
    # print(f"Total successful altered queries: {total_successful_altered_queries}/{total_queries}")


if __name__ == "__main__":
    main(sys.argv[1:])
