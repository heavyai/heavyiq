import argparse
import heavyai
import pandas as pd
import sys


def getOptions(argv=None):
    parser = argparse.ArgumentParser(description="Find empty tables in HeavyDB test data")
    parser.add_argument("-s", "--host", help="HeavyDB server address", default="localhost")
    parser.add_argument("-p", "--port", help="HeavyDB server port", default="6273")
    parser.add_argument("-u", "--user", help="HeavyDB user name", default="admin")
    parser.add_argument("-w", "--password", help="HeavyDB password", default="HyperInteractive")
    parser.add_argument("-q", "--queries", help="Queries CSV file", default="./spider_qa.csv")
    return parser.parse_args(argv)


def getQueriesByDB(queries_file):
    queries_df = pd.read_csv(queries_file)
    queries_df["modified_sql_query"] = queries_df["modified_sql_query"].astype(str)
    queries_by_db = {}
    for index, row in queries_df.iterrows():
        query_id = row["query_id"]
        db_id = row["db_id"]
        original_sql_query = row["original_sql_query"].strip()
        modified_sql_query = row["modified_sql_query"].strip()
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
                }
            )
    return queries_by_db


def main(argv):
    options = getOptions(argv)
    queries_by_db = getQueriesByDB(options.queries)
    empty_dbs = []
    partially_empty_dbs = []
    populated_dbs = []
    for db_id, queries in queries_by_db.items():
        con = None
        try:
            con = heavyai.connect(user=options.user, password=options.password, host=options.host, dbname=db_id)
        except Exception as e:
            print(f"Error connecting to database {db_id}: {e}")
            continue
        db_tables = con.get_tables()
        all_empty_tables = True
        has_empty_table = False
        for db_table in db_tables:
            res = con.execute(f"SELECT COUNT(*) FROM {db_table}")
            num_rows = list(res)[0][0]
            if num_rows > 0:
                all_empty_tables = False
            else:
                has_empty_table = True
        if all_empty_tables:
            empty_dbs.append(db_id)
        elif has_empty_table:
            partially_empty_dbs.append(db_id)
        else:
            populated_dbs.append(db_id)
    print(f"Empty DBs: {empty_dbs}")
    print(f"Partially empty DBs: {partially_empty_dbs}")


if __name__ == "__main__":
    main(sys.argv[1:])
