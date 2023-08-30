import argparse
import heavyai
import sys


def getOptions(argv=None):
    parser = argparse.ArgumentParser(description="Build HeavyIQ training dataset")
    parser.add_argument("-s", "--host", help="HeavyDB server address", default="localhost")
    parser.add_argument("-p", "--port", help="HeavyDB server port", default="6273")
    parser.add_argument("-u", "--user", help="HeavyDB user name", default="admin")
    parser.add_argument("-w", "--password", help="HeavyDB password", default="HyperInteractive")
    parser.add_argument("-d", "--database", help="HeavyDB database", default="heavyai")
    parser.add_argument("-v", "--version", help="Dataset Version Num", default=None)
    parser.add_argument("-q", "--queries", help="Queries file", default=None)
    parser.add_argument("-t", "--max-token-length", help="Max token length", default=768)
    parser.add_argument("-o", "--output-dir", help="Output directory", default=None)
    return parser.parse_args(argv)


def load_queries(con, queries_path, max_token_length, version, output_dir):
    drop_table_sql = f"DROP TABLE IF EXISTS heavyiq_text_to_sql_v{version};"
    create_table_sql = f"CREATE TABLE heavyiq_text_to_sql_v{version}_{max_token_length}_tokens (query_id INT, db_id TEXT, tables TEXT[], data_split TEXT, num_instruction_tokens INT, num_targeted_instruction_tokens INT, num_output_tokens INT, instruction TEXT, targeted_instruction TEXT, output TEXT);"
    load_sql = f"COPY heavyiq_text_to_sql_v{version}_{max_token_length}_tokens FROM '{queries_path}' WITH (header='t', array_marker='[]');"
    export_train_sql = f"COPY (SELECT query_id, db_id, targeted_instruction AS instruction, output FROM heavyiq_text_to_sql_v{version}_{max_token_length}_tokens WHERE data_split <> 'dev' AND num_targeted_instruction_tokens <= {max_token_length}) TO '{output_dir}/heavyiq_text_to_sql_v{version}_{max_token_length}_tokens_train.csv' WITH (header='t');"
    export_eval_sql = f"COPY (SELECT query_id, db_id, targeted_instruction AS instruction, output FROM heavyiq_text_to_sql_v{version}_{max_token_length}_tokens WHERE data_split = 'dev' AND num_targeted_instruction_tokens <= {max_token_length}) TO '{output_dir}/heavyiq_text_to_sql_v{version}_{max_token_length}_tokens_eval.csv' WITH (header='t');"
    con.execute(drop_table_sql)
    con.execute(create_table_sql)
    con.execute(load_sql)
    con.execute(export_train_sql)
    con.execute(export_eval_sql)


def main(argv):
    options = getOptions(argv)
    con = heavyai.connect(user=options.user, password=options.password, host=options.host, dbname=options.database)
    if options.queries is not None:
        load_queries(con, options.queries, options.max_token_length, options.version, options.output_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
