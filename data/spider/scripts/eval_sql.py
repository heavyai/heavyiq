import argparse
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


def main(argv):
    options = getOptions(argv)
    queries_by_db = getQueriesByDB(options.queries)
    con = None
    for db_id, queries in queries_by_db.items():
        try:
            con = heavyai.connect(user=options.user, password=options.password, host=options.host, dbname=db_id)
        except Exception as e:
            print(f"Error connecting to database {db_id}: {e}")
            continue
        num_queries = len(queries)
        for query in queries:
            query_id = query["query_id"]
            gold_sql_query = query["gold_sql_query"]
            gen_sql_query = query["gen_sql_query"]
            gold_result = None
            gen_result = None
            try:
                gold_result = con.select_ipc(gold_sql_query)
            except Exception as e:
                print(f"Error executing gold query {query_id} on database {db_id}: {e}")
                continue
            try:
                gen_result = con.select_ipc(gen_sql_query)
            except Exception as e:
                print(f"Error executing generated query {query_id} on database {db_id}: {e}")
                continue


if __name__ == "__main__":
    main(sys.argv[1:])
