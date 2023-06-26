import argparse
import csv
import heavyai
import json
import pandas as pd
import re
import sys


def getOptions(argv=None):
    parser = argparse.ArgumentParser(description="Test Spider SQL queries against HeavyDB")
    parser.add_argument("-s", "--host", help="HeavyDB server address", default="localhost")
    parser.add_argument("-p", "--port", help="HeavyDB server port", default="6273")
    parser.add_argument("-u", "--user", help="HeavyDB user name", default="admin")
    parser.add_argument("-w", "--password", help="HeavyDB password", default="HyperInteractive")
    parser.add_argument("-q", "--queries", help="Queries CSV file", default="./spider_qa.csv")
    parser.add_argument("-d", "--database", help="HeavyDB database", default="None")
    parser.add_argument("--start-db", help="Start from this database idx", type=int, default=None)
    parser.add_argument("--num-dbs", help="Only process this many databases", type=int, default=0)
    parser.add_argument("--write-prompts", help="Write prompts to file for successful queries", action="store_true")
    return parser.parse_args(argv)


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
    fieldnames = ["instruction", "output"]
    prompts_to_write = [(obj["instruction"], obj["output"]) for obj in prompts]
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(fieldnames)
        writer.writerows(prompts_to_write)


def main(argv):
    options = getOptions(argv)
    queries_by_db = getQueriesByDB(options.queries)
    db_num = 0
    total_queries = 0
    total_successful_queries = 0
    total_successful_altered_queries = 0
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
                sql_query = re.sub(" +", " ", sql_query)
                sql_query = re.sub(" ,", ",", sql_query)
                sql_query = sql_query + ";" if sql_query[-1] != ";" else sql_query

                # print(f"Query ID: {query_id}")
                # print(f"SQL Query: {sql_query}")
                try:
                    results = list(con.execute(sql_query))
                    successful_queries += 1
                    if options.write_prompts:
                        instruction = generate_instruction(table_schemas, query["question"])
                        # sql_query_with_semicolon = sql_query + ";" if sql_query[-1] != ";" else sql_query
                        prompts.append({"instruction": instruction, "input": "", "output": sql_query})
                except Exception as e:
                    if '"' in sql_query:
                        sql_query = sql_query.replace('"', "'")
                        try:
                            results = list(con.execute(sql_query))
                            successful_queries += 1
                            total_successful_altered_queries += 1
                        except Exception as e:
                            failed_queries.append({"query_id": query_id, "db_id": db_id, "sql_query": sql_query})
                            continue
                    else:
                        # failed_queries.append({"query_id": query_id, "db_id": db_id, "sql_query": sql_query, "error": e})
                        failed_queries.append({"query_id": query_id, "db_id": db_id, "sql_query": sql_query})
                        continue
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
    if options.write_prompts:
        with open("sql_training_prompts.json", "w") as file:
            json.dump(prompts, file, indent=4)
        write_prompts_to_csv(prompts, "sql_training_prompts.csv")
    print(f"\n\nTotal successful queries: {total_successful_queries}/{total_queries}")
    print(f"Total successful altered queries: {total_successful_altered_queries}/{total_queries}")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        pass
