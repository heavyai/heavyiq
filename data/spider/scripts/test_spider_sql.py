import argparse
import heavyai
import pandas as pd
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
        if db_id not in queries_by_db:
            queries_by_db[db_id] = [
                {
                    "query_id": query_id,
                    "original_sql_query": original_sql_query,
                    "modified_sql_query": modified_sql_query,
                }
            ]
        else:
            queries_by_db[db_id].append(
                {
                    "query_id": query_id,
                    "original_sql_query": original_sql_query,
                    "modified_sql_query": modified_sql_query,
                }
            )
    return queries_by_db


def main(argv):
    options = getOptions(argv)
    queries_by_db = getQueriesByDB(options.queries)
    db_num = 0
    total_queries = 0
    total_successful_queries = 0
    total_successful_altered_queries = 0
    failed_queries = []
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
            for query in queries:
                query_id = query["query_id"]
                original_sql_query = query["original_sql_query"]
                modified_sql_query = query["modified_sql_query"]
                sql_query = modified_sql_query if len(str(modified_sql_query)) > 3 else original_sql_query
                # if '"' in sql_query:
                #    sql_query = sql_query.replace('"', "'")
                # print(f"Query ID: {query_id}")
                # print(f"SQL Query: {sql_query}")
                try:
                    results = list(con.execute(sql_query))
                    successful_queries += 1
                    # print(f"Results: {results}")
                except Exception as e:
                    if '"' in sql_query:
                        sql_query = sql_query.replace('"', "'")
                        try:
                            # print(sql_query)
                            results = list(con.execute(sql_query))
                            successful_queries += 1
                            total_successful_altered_queries += 1
                        except Exception as e:
                            failed_queries.append({"query_id": query_id, "db_id": db_id, "sql_query": sql_query})
                            continue
                    else:
                        # failed_queries.append({"query_id": query_id, "db_id": db_id, "sql_query": sql_query, "error": e})
                        failed_queries.append({"query_id": query_id, "db_id": db_id, "sql_query": sql_query})
                        # print(f"Error: {e}")
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
    print(f"\n\nTotal successful queries: {total_successful_queries}/{total_queries}")
    print(f"Total successful altered queries: {total_successful_altered_queries}/{total_queries}")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        pass
