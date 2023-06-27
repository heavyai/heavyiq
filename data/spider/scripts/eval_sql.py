import argparse
import csv
import heavyai
import sys


def getQueriesByDB(queries_file):
    queries_df = pd.read_csv(queries_file)
    queries_df["question"] = queries_df["question"].astype(str)
    queries_df["gold_sql_query"] = queries_df["gold_sql_query"].astype(str)
    queries_df["gen_sql_query"] = queries_df["gen_sql_query"].astype(str)
    queries_by_db = {}
    for index, row in queries_df.iterrows():
        query_id = row["query_id"]
        db_id = row["db_id"]
        gold_sql_query = row["gold_sql_query"]
        gen_sql_query = row["gen_sql_query"]
        question = row["question"]
        if db_id not in queries_by_db:
            queries_by_db[db_id] = [
                {
                    "query_id": query_id,
                    "question": question,
                    "gold_sql_query": gold_sql_query,
                    "gen_sql_query": gen_sql_query,
                }
            ]
        else:
            queries_by_db[db_id].append(
                {
                    "query_id": query_id,
                    "question": question,
                    "gold_sql_query": gold_sql_query,
                    "gen_sql_query": gen_sql_query,
                }
            )
    return queries_by_db


def getOptions(argv=None):
    parser = argparse.ArgumentParser(description="Eval SQL queries against ground truth queries in HeavyDB")
    parser.add_argument("-s", "--host", help="HeavyDB server address", default="localhost")
    parser.add_argument("-p", "--port", help="HeavyDB server port", default="6273")
    parser.add_argument("-u", "--user", help="HeavyDB user name", default="admin")
    parser.add_argument("-w", "--password", help="HeavyDB password", default="HyperInteractive")
    parser.add_argument("--queries", help="Test queries CSV file")
    # parser.add_argument("--gold-queries", help="Gold eval queries CSV file")
    return parser.parse_args(argv)


def writeQueryResultsCsv(filename, query_results):
    # Specify the fieldnames (headers of the CSV)
    fieldnames = ["query_id", "db_id", "question", "gold_sql_query", "gen_sql_query", "gold_run_success", "gen_result"]

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
    queries_by_db = getQueriesByDB(options.queries)
    con = None
    query_results = []
    for db_id, queries in queries_by_db.items():
        try:
            con = heavyai.connect(user=options.user, password=options.password, host=options.host, dbname=db_id)
        except Exception as e:
            print(f"Error connecting to database {db_id}: {e}")
            continue
        num_queries = len(queries)
        for query in queries:
            query_id = query["query_id"]
            question = query["question"]
            gold_sql_query = query["gold_sql_query"]
            gen_sql_query = query["gen_sql_query"]
            gold_result = None
            gen_result = None
            gold_run_success = False
            gen_run_success = False
            try:
                gold_result = con.select_ipc(gold_sql_query)
                gold_run_success = True
            except Exception as e:
                print(f"Error executing gold query {query_id} on database {db_id}: {e}")
                continue
            try:
                gen_result = con.select_ipc(gen_sql_query)
                gen_run_success = True
            except Exception as e:
                print(f"Error executing generated query {query_id} on database {db_id}: {e}")
                continue
            query_results.append(
                {
                    "query_id": query_id,
                    "db_id": db_id,
                    "question": question,
                    "gold_sql_query": gold_sql_query,
                    "gen_sql_query": gen_sql_query,
                    "gold_run_success": gold_run_success,
                    "gen_result": gen_run_success,
                }
            )
    writeQueryResultsCsv("eval_results.csv", query_results)


if __name__ == "__main__":
    main(sys.argv[1:])
