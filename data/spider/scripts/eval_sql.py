import argparse
import csv
import heavyai
import json
import pandas as pd
import re
import sys


def getQueriesByDB(prompts_filename):
    prompts = []
    with open(prompts_filename, "r") as f:
        prompts = json.load(f)
    queries_by_db = {}
    for prompt in prompts:
        db_id = prompt["db_id"]
        if db_id not in queries_by_db:
            queries_by_db[db_id] = [prompt]
        else:
            queries_by_db[db_id].append(prompt)
    return queries_by_db


def getOptions(argv=None):
    parser = argparse.ArgumentParser(description="Eval SQL queries against ground truth queries in HeavyDB")
    parser.add_argument("-s", "--host", help="HeavyDB server address", default="localhost")
    parser.add_argument("-p", "--port", help="HeavyDB server port", default="6273")
    parser.add_argument("-u", "--user", help="HeavyDB user name", default="admin")
    parser.add_argument("-w", "--password", help="HeavyDB password", default="HyperInteractive")
    parser.add_argument("--prompts", help="Test prompts JSON file")
    # parser.add_argument("--gold-queries", help="Gold eval queries CSV file")
    return parser.parse_args(argv)


def writeQueryResultsCsv(filename, query_results):
    # Specify the fieldnames (headers of the CSV)
    fieldnames = [
        "query_id",
        "db_id",
        "prompt",
        "gold_sql_query",
        "gen_sql_query",
        "gold_run_success",
        "gen_run_success",
        "gold_row_count",
        "gold_col_count",
        "gen_row_count",
        "gen_col_count",
        "gold_gen_match",
    ]

    # Write the dictionaries to a CSV file
    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header row
        writer.writeheader()

        # Write all query results
        for query_result in query_results:
            writer.writerow(query_result)


def main(argv):
    options = getOptions(argv)
    queries_by_db = getQueriesByDB(options.prompts)
    con = None
    query_results = []
    query_idx = 0
    max_queries = 700
    for db_id, queries in queries_by_db.items():
        if query_idx >= max_queries:
            break
        try:
            con = heavyai.connect(user=options.user, password=options.password, host=options.host, dbname=db_id)
        except Exception as e:
            print(f"Error connecting to database {db_id}: {e}")
            continue
        num_queries = len(queries)
        for query in queries:
            if query_idx >= max_queries:
                break
            query_id = query["query_id"]
            prompt = query["prompt"]
            gold_sql_query = query["gold_sql_query"]
            eval_sql_query = query["gen_sql_query"]
            # if not gen_sql_query.lower().startswith("select"):
            #    match = re.search(r"(SELECT.*FROM.*$)", gen_sql_query, re.DOTALL | re.MULTILINE | re.IGNORECASE)
            #    if match:
            #        gen_sql_query = match.group(1)
            # if not gen_sql_query.endswith(";"):
            #    gen_sql_query += ";"
            gold_result = None
            eval_result = None
            gold_run_success = False
            eval_run_success = False
            gold_row_count = None
            gold_col_count = None
            eval_row_count = None
            eval_col_count = None
            gold_eval_match = False
            try:
                gold_df = pd.read_sql(gold_sql_query)
                gold_run_success = True
                num_gold_rows = len(gold_df.axes[0])
                num_gold_cols = len(gold_df.axes[1])
            except Exception as e:
                print(f"Error executing gold query {query_id} on database {db_id}: {e}")
            try:
                eval_df = con.execute(eval_sql_query)
                eval_run_success = True
                num_eval_rows = len(eval_df.axes[0])
                num_eval_cols = len(eval_df.axes[1])
            except Exception as e:
                print(f"Error executing generated query {query_id} on database {db_id}: {e}")
                print(gen_sql_query)
            if (
                gold_run_success
                and eval_run_success
                and num_gold_rows == num_eval_rows
                and num_gold_cols == num_eval_cols
            ):
                gold_gen_match = True
            query_results.append(
                {
                    "query_id": query_id,
                    "db_id": db_id,
                    "prompt": prompt,
                    "gold_sql_query": gold_sql_query,
                    "gen_sql_query": gen_sql_query,
                    "gold_run_success": gold_run_success,
                    "gen_run_success": gen_run_success,
                    "gold_row_count": gold_row_count,
                    "gold_col_count": gold_col_count,
                    "gen_row_count": gen_row_count,
                    "gen_col_count": gen_col_count,
                    "gold_gen_match": gold_gen_match,
                }
            )
            print(query_idx)
            query_idx += 1
    writeQueryResultsCsv("eval_results.csv", query_results)


if __name__ == "__main__":
    main(sys.argv[1:])
