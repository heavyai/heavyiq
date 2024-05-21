import argparse
import csv
import os
import pandas as pd
import random
import shutil
import sys


def getOptions(argv=None):
    parser = argparse.ArgumentParser(description="Split SQL queries into training and eval datasets")
    parser.add_argument("-i", "--input", help="Input CSV file", default="")
    parser.add_argument("-o", "--output", help="Output Folder", default="")
    parser.add_argument("-q", "--queries-eval", help="Number of target queries in eval dataset", default=10)
    return parser.parse_args(argv)


def getQueriesByDB(queries_file):
    queries_df = pd.read_csv(queries_file)
    queries_by_db = {}
    for index, row in queries_df.iterrows():
        db_id = row["db_id"]
        row_dict = row.to_dict()
        if db_id not in queries_by_db:
            queries_by_db[db_id] = [row_dict]
        else:
            queries_by_db[db_id].append(row_dict)
    return queries_by_db


def write_train_eval_csvs(file_prefix, queries_by_db, eval_dbs):
    train_queries = []
    eval_queries = []
    print(f"Eval DBs: {', '.join(eval_dbs)}")
    for db in queries_by_db:
        if db in eval_dbs:
            for query in queries_by_db[db]:
                eval_queries.append(query)
        else:
            for query in queries_by_db[db]:
                train_queries.append(query)
    train_filename = file_prefix + "train.csv"
    eval_filename = file_prefix + "eval.csv"
    csv_fields = train_queries[0].keys()
    # csv_fields = obj.keys()
    # print(csv_fields)
    # Create a DictWriter instance
    with open(train_filename, "w", newline="") as train_file:
        train_writer = csv.DictWriter(train_file, fieldnames=csv_fields)
        train_writer.writeheader()
        for train_query in train_queries:
            train_writer.writerow(train_query)

    with open(eval_filename, "w", newline="") as eval_file:
        eval_writer = csv.DictWriter(eval_file, fieldnames=csv_fields)
        eval_writer.writeheader()
        for eval_query in eval_queries:
            eval_writer.writerow(eval_query)


def main(argv):
    options = getOptions(argv)
    input_csv = options.input
    output_folder = options.output
    if os.path.exists(output_folder) and os.path.isdir(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    queries_by_db = getQueriesByDB(input_csv)
    db_names = list(queries_by_db.keys())
    random.seed(42)
    random.shuffle(db_names)
    num_dbs = len(db_names)
    print(db_names)
    print(f"Num DBS: {num_dbs}")
    # num_dbs_eval = int(options.num_dbs_eval)
    # for start_idx in range(0, num_dbs, num_dbs_eval):
    target_queries_eval = int(options.queries_eval)
    num_eval_queries = 0
    eval_dbs = []
    train_eval_idx = 1
    for db_name in db_names:
        num_db_queries = len(queries_by_db[db_name])
        # print(f"Num DB queries: {num_db_queries}")
        # print(f"Num Eval queries: {num_eval_queries}")
        if num_eval_queries + num_db_queries > target_queries_eval and num_eval_queries > 0:
            file_prefix = f"{output_folder}/sql_{train_eval_idx}_"
            write_train_eval_csvs(file_prefix, queries_by_db, eval_dbs)
            train_eval_idx += 1
            num_eval_queries = num_db_queries
            eval_dbs = [db_name]
        else:
            num_eval_queries += num_db_queries
            eval_dbs.append(db_name)
    if len(eval_dbs) > 0:
        file_prefix = f"{output_folder}/sql_{train_eval_idx}_"
        write_train_eval_csvs(file_prefix, queries_by_db, eval_dbs)


if __name__ == "__main__":
    main(sys.argv[1:])
