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
    parser.add_argument("--query-instruction-prompt", help="Query instruction prompt", default=None)
    parser.add_argument("--query-output-prompt", help="Query output prompt", default=None)
    parser.add_argument("--nl-answers", help="NL Answers file", default=None)
    parser.add_argument("--nl-answers-instruction-prompt", help="NL Answers instruction prompt", default=None)
    parser.add_argument("--nl-answers-output-prompt", help="NL Answers output prompt", default=None)
    parser.add_argument("-t", "--max-token-length", help="Max token length", default=768)
    parser.add_argument("-o", "--output-dir", help="Output directory", default=None)
    parser.add_argument("--create-combo-dataset", help="Create combo dataset", action="store_true")
    parser.add_argument("--instruction-table", help="NL Answers file", default="databricks_dolly_15k")

    return parser.parse_args(argv)


def load_queries(
    con, queries_path, query_instruction_prompt, query_output_prompt, max_token_length, version, output_dir
):
    drop_table_sql = f"DROP TABLE IF EXISTS heavyiq_text_to_sql_v{version}_{max_token_length}_tokens;"
    create_table_sql = f"CREATE TABLE heavyiq_text_to_sql_v{version}_{max_token_length}_tokens (query_id INT, db_id TEXT, tables TEXT[], data_split TEXT, num_instruction_tokens INT, num_targeted_instruction_tokens INT, num_output_tokens INT, instruction TEXT, targeted_instruction TEXT, output TEXT);"
    load_sql = f"COPY heavyiq_text_to_sql_v{version}_{max_token_length}_tokens FROM '{queries_path}' WITH (header='t', array_marker='[]');"
    add_prompt_col_sql = f"ALTER TABLE heavyiq_text_to_sql_v{version}_{max_token_length}_tokens ADD COLUMN prompt TEXT;"
    add_answer_col_sql = f"ALTER TABLE heavyiq_text_to_sql_v{version}_{max_token_length}_tokens ADD COLUMN answer TEXT;"
    update_prompt_col_sql = f"UPDATE heavyiq_text_to_sql_v{version}_{max_token_length}_tokens SET prompt = instruction;"
    if query_instruction_prompt and query_output_prompt is not None:
        update_prompt_col_sql = f"UPDATE heavyiq_text_to_sql_v{version}_{max_token_length}_tokens SET prompt = '{query_instruction_prompt}\n' || instruction || '\n{query_output_prompt}\n';"
    update_answer_col_sql = f"UPDATE heavyiq_text_to_sql_v{version}_{max_token_length}_tokens SET answer = output;"
    export_train_sql = f"COPY (SELECT query_id, db_id, prompt, answer FROM heavyiq_text_to_sql_v{version}_{max_token_length}_tokens WHERE data_split <> 'dev' AND num_targeted_instruction_tokens <= {max_token_length}) TO '{output_dir}/heavyiq_text_to_sql_v{version}_{max_token_length}_tokens_train.csv' WITH (header='t');"
    export_eval_sql = f"COPY (SELECT query_id, db_id, prompt, answer FROM heavyiq_text_to_sql_v{version}_{max_token_length}_tokens WHERE data_split = 'dev' AND num_targeted_instruction_tokens <= {max_token_length}) TO '{output_dir}/heavyiq_text_to_sql_v{version}_{max_token_length}_tokens_eval.csv' WITH (header='t');"

    con.execute(drop_table_sql)
    con.execute(create_table_sql)
    con.execute(load_sql)
    con.execute(add_prompt_col_sql)
    con.execute(add_answer_col_sql)
    con.execute(update_prompt_col_sql)
    con.execute(update_answer_col_sql)
    con.execute(export_train_sql)
    con.execute(export_eval_sql)


def load_nl_answers(con, nl_answers_path, nl_answers_instruction_prompt, nl_answers_output_prompt, version, output_dir):
    drop_table_sql = f"DROP TABLE IF EXISTS heavyiq_nl_answers_v{version}"
    create_table_sql = f"CREATE TABLE heavyiq_nl_answers_v{version} (query_id INT, db_id TEXT, data_split TEXT, instruction TEXT, output TEXT);"
    load_sql = f"COPY heavyiq_nl_answers_v{version} FROM '{nl_answers_path}' WITH (header='t');"
    add_prompt_col_sql = f"ALTER TABLE heavyiq_nl_answers_v{version} ADD COLUMN prompt TEXT;"
    add_answer_col_sql = f"ALTER TABLE heavyiq_nl_answers_v{version} ADD COLUMN answer TEXT;"
    update_prompt_col_sql = f"UPDATE heavyiq_nl_answers_v{version} SET prompt = instruction;"
    if nl_answers_instruction_prompt and nl_answers_output_prompt is not None:
        update_prompt_col_sql = f"UPDATE heavyiq_nl_answers_v{version} SET prompt = '{nl_answers_instruction_prompt}\n' || instruction || '\n{nl_answers_output_prompt}\n';"
    update_answer_col_sql = f"UPDATE heavyiq_nl_answers_v{version} SET answer = output;"
    export_train_sql = f"COPY (SELECT query_id, db_id, prompt, answer FROM heavyiq_nl_answers_v{version} WHERE SAMPLE_RATIO(0.9)) TO '{output_dir}/heavyiq_nl_answers_v{version}_train.csv' WITH (header='t');"
    export_eval_sql = f"COPY (SELECT query_id, db_id, prompt, answer FROM heavyiq_nl_answers_v{version} WHERE NOT SAMPLE_RATIO(0.9)) TO '{output_dir}/heavyiq_nl_answers_v{version}_eval.csv' WITH (header='t');"

    con.execute(drop_table_sql)
    con.execute(create_table_sql)
    con.execute(load_sql)
    con.execute(add_prompt_col_sql)
    con.execute(add_answer_col_sql)
    con.execute(update_prompt_col_sql)
    con.execute(update_answer_col_sql)
    con.execute(export_train_sql)
    con.execute(export_eval_sql)


def create_combo_dataset(con, version, instruction_table, sql_table, nl_answers_table, only_sql_in_eval, output_dir):
    assert only_sql_in_eval, "only_sql_in_eval must be True"

    drop_table_sql = f"DROP TABLE IF EXISTS heavyiq_combo_v{version}"
    create_table_sql = f"CREATE TABLE heavyiq_combo_v{version} (id INT, data_split TEXT, prompt TEXT, answer TEXT);"
    load_sql_sql = f"INSERT INTO heavyiq_combo_v{version} SELECT query_id, CASE WHEN data_split <> 'dev' THEN 'train' ELSE 'eval' END, prompt, answer FROM {sql_table};"
    load_answers_sql = f"INSERT INTO heavyiq_combo_v{version} SELECT query_id + 1000000, 'train', prompt, answer FROM {nl_answers_table};"
    load_dolly_sql = f"INSERT INTO heavyiq_combo_v{version} SELECT id + 2000000, 'train', prompt, answer FROM {instruction_table} WHERE length(prompt) < 1760 AND length(answer) < 640;"
    export_train_sql = f"COPY (SELECT id, prompt, answer FROM heavyiq_combo_v{version} WHERE data_split = 'train') TO '{output_dir}/heavyiq_combo_v{version}_train.csv' WITH (header='t');"
    export_eval_sql = f"COPY (SELECT id, prompt, answer FROM heavyiq_combo_v{version} WHERE data_split = 'eval') TO '{output_dir}/heavyiq_combo_v{version}_eval.csv' WITH (header='t');"

    con.execute(drop_table_sql)
    con.execute(create_table_sql)
    con.execute(load_sql_sql)
    con.execute(load_answers_sql)
    con.execute(load_dolly_sql)
    con.execute(export_train_sql)
    con.execute(export_eval_sql)


def main(argv):
    options = getOptions(argv)
    con = heavyai.connect(user=options.user, password=options.password, host=options.host, dbname=options.database)
    if options.queries is not None:
        load_queries(
            con,
            options.queries,
            options.query_instruction_prompt,
            options.query_output_prompt,
            options.max_token_length,
            options.version,
            options.output_dir,
        )
    if options.nl_answers is not None:
        load_nl_answers(
            con,
            options.nl_answers,
            options.nl_answers_instruction_prompt,
            options.nl_answers_output_prompt,
            options.version,
            options.output_dir,
        )
    if options.create_combo_dataset and options.queries and options.nl_answers:
        create_combo_dataset(
            con,
            options.version,
            options.instruction_table,
            f"heavyiq_text_to_sql_v{options.version}_{options.max_token_length}_tokens",
            f"heavyiq_nl_answers_v{options.version}",
            True,
            options.output_dir,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
