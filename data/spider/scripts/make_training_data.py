import argparse
import heavyai
import sys


def getOptions(argv=None):
    parser = argparse.ArgumentParser(description="Build HeavyIQ training dataset")
    parser.add_argument(
        "-s", "--host", help="HeavyDB server address", default="localhost"
    )
    parser.add_argument("-p", "--port", help="HeavyDB server port", default="6273")
    parser.add_argument("-u", "--user", help="HeavyDB user name", default="admin")
    parser.add_argument(
        "-w", "--password", help="HeavyDB password", default="HyperInteractive"
    )
    parser.add_argument("-d", "--database", help="HeavyDB database", default="heavyai")
    parser.add_argument("-v", "--version", help="Dataset Version Num", default=None)
    parser.add_argument("-q", "--queries", help="Queries file", default=None)
    parser.add_argument(
        "--num-query-shards", help="Number of query shards", default=None, type=int
    )
    parser.add_argument(
        "--query-instruction-prompt", help="Query instruction prompt", default=None
    )
    parser.add_argument(
        "--query-output-prompt", help="Query output prompt", default=None
    )
    parser.add_argument("--nl-answers", help="NL Answers file", default=None)
    parser.add_argument(
        "--nl-answers-instruction-prompt",
        help="NL Answers instruction prompt",
        default=None,
    )
    parser.add_argument(
        "--nl-answers-output-prompt", help="NL Answers output prompt", default=None
    )
    parser.add_argument(
        "--questions-table-level", help="Table questions file", default=None
    )
    parser.add_argument(
        "--questions-table-level-instruction-prompt",
        help="Table questions instruction prompt",
        default=None,
    )
    parser.add_argument(
        "--questions-table-level-output-prompt",
        help="Table questions output prompt",
        default=None,
    )
    parser.add_argument(
        "--questions-column-level", help="Column questions file", default=None
    )
    parser.add_argument(
        "--questions-column-level-instruction-prompt",
        help="Column questions instruction prompt",
        default=None,
    )
    parser.add_argument(
        "--questions-column-level-output-prompt",
        help="Column questions output prompt",
        default=None,
    )
    parser.add_argument("--used-tables", help="Used tables file", default=None)
    parser.add_argument(
        "--used-tables-instruction-prompt",
        help="Used tables instruction prompt",
        default=None,
    )
    parser.add_argument(
        "--used-tables-output-prompt", help="Used tables output prompt", default=None
    )
    parser.add_argument("-e", "--errors", help="Errors file", default=None)
    parser.add_argument(
        "--errors-instruction-prompt", help="Errors instruction prompt", default=None
    )
    parser.add_argument(
        "--errors-output-prompt", help="Errors output prompt", default=None
    )
    parser.add_argument(
        "-t", "--max-token-length", help="Max token length", default=768
    )
    parser.add_argument("-o", "--output-dir", help="Output directory", default=None)
    parser.add_argument(
        "--create-combo-dataset", help="Create combo dataset", action="store_true"
    )
    parser.add_argument(
        "--instruction-table", help="NL Answers file", default="databricks_dolly_15k"
    )
    parser.add_argument("--label", help="Label", default=None)
    parser.add_argument("--combo-label", help="Combo label", default=None)
    parser.add_argument("--query-label", help="Query label", default=None)
    parser.add_argument("--min-avg-prob", help="Min avg prob", default=None)
    parser.add_argument(
        "--avg-prob-order-noise", help="Order by avg prob noise", default=None
    )

    return parser.parse_args(argv)


def load_queries(
    con,
    queries_path,
    query_instruction_prompt,
    query_output_prompt,
    max_token_length,
    label,
    version,
    output_dir,
    num_shards,
    min_avg_prob,
    order_by_avg_prob_noise,
):
    query_table_label = (
        f"v{version}_{max_token_length}_tokens_{label}"
        if label is not None
        else f"v{version}_{max_token_length}_tokens"
    )

    drop_table_sql = f"DROP TABLE IF EXISTS heavyiq_text_to_sql_{query_table_label};"
    create_table_sql = f"CREATE TABLE heavyiq_text_to_sql_{query_table_label} (query_id INT, db_id TEXT, tables TEXT[], data_split TEXT, question TEXT, num_instruction_tokens INT, num_targeted_instruction_tokens INT, num_output_tokens INT, instruction TEXT, targeted_instruction TEXT, output TEXT, avg_prob DOUBLE, total_prob DOUBLE, min_prop DOUBLE);"
    load_sql = f"COPY heavyiq_text_to_sql_{query_table_label} FROM '{queries_path}' WITH (header='t', array_marker='[]');"
    add_prompt_col_sql = (
        f"ALTER TABLE heavyiq_text_to_sql_{query_table_label} ADD COLUMN prompt TEXT;"
    )
    add_answer_col_sql = (
        f"ALTER TABLE heavyiq_text_to_sql_{query_table_label} ADD COLUMN answer TEXT;"
    )
    update_prompt_col_sql = f"UPDATE heavyiq_text_to_sql_{query_table_label} SET prompt = targeted_instruction;"
    if query_instruction_prompt and query_output_prompt is not None:
        update_prompt_col_sql = f"UPDATE heavyiq_text_to_sql_{query_table_label} SET prompt = '{query_instruction_prompt}\n' || targeted_instruction || '\n{query_output_prompt}\n';"
    update_answer_col_sql = (
        f"UPDATE heavyiq_text_to_sql_{query_table_label} SET answer = output;"
    )

    export_prefix = f"heavyiq_text_to_sql_{query_table_label}"
    avg_prob_filter_clause = ""
    if min_avg_prob is not None:
        avg_prob_filter_clause = f"AND avg_prob >= {min_avg_prob}"
    avg_prob_order_by_clause = ""
    if order_by_avg_prob_noise is not None:
        avg_prob_order_by_clause = f"ORDER BY avg_prob + mod(hash(prompt), 100) * 0.01 * {order_by_avg_prob_noise} DESC"

    export_train_sql = f"COPY (SELECT query_id, db_id, question, prompt, answer FROM heavyiq_text_to_sql_{query_table_label} WHERE data_split <> 'dev' AND num_targeted_instruction_tokens <= {max_token_length} {avg_prob_filter_clause} {avg_prob_order_by_clause}) TO '{output_dir}/{export_prefix}_train.csv' WITH (header='t');"
    print(export_train_sql)
    export_eval_sql = f"COPY (SELECT query_id, db_id, question, prompt, answer FROM heavyiq_text_to_sql_{query_table_label} WHERE data_split = 'dev' AND num_targeted_instruction_tokens <= {max_token_length}) TO '{output_dir}/{export_prefix}_eval.csv' WITH (header='t');"

    con.execute(drop_table_sql)
    con.execute(create_table_sql)
    con.execute(load_sql)
    con.execute(add_prompt_col_sql)
    con.execute(add_answer_col_sql)
    con.execute(update_prompt_col_sql)
    con.execute(update_answer_col_sql)
    con.execute(export_train_sql)
    con.execute(export_eval_sql)

    if num_shards is not None:
        for excluded_shard in range(num_shards):
            export_shard_train_sql = f"COPY (SELECT query_id, db_id, prompt, answer FROM heavyiq_text_to_sql_v{version}_{max_token_length}_tokens WHERE MOD(ABS(HASH(db_id)), {num_shards}) <> {excluded_shard} AND num_targeted_instruction_tokens <= {max_token_length}) TO '{output_dir}/heavyiq_text_to_sql_v{version}_{max_token_length}_tokens_no_shard_{excluded_shard}_train.csv' WITH (header='t');"
            export_shard_eval_sql = f"COPY (SELECT query_id, db_id, prompt, answer FROM heavyiq_text_to_sql_v{version}_{max_token_length}_tokens WHERE MOD(ABS(HASH(db_id)), {num_shards}) = {excluded_shard} AND num_targeted_instruction_tokens <= {max_token_length}) TO '{output_dir}/heavyiq_text_to_sql_v{version}_{max_token_length}_tokens_no_shard_{excluded_shard}_eval.csv' WITH (header='t');"
            con.execute(export_shard_train_sql)
            con.execute(export_shard_eval_sql)
    return f"heavyiq_text_to_sql_{query_table_label}"


def load_nl_answers(
    con,
    nl_answers_path,
    nl_answers_instruction_prompt,
    nl_answers_output_prompt,
    version,
    output_dir,
):
    drop_table_sql = f"DROP TABLE IF EXISTS heavyiq_nl_answers_v{version}"
    create_table_sql = f"CREATE TABLE heavyiq_nl_answers_v{version} (query_id INT, db_id TEXT, data_split TEXT, instruction TEXT, output TEXT);"
    load_sql = f"COPY heavyiq_nl_answers_v{version} FROM '{nl_answers_path}' WITH (header='t');"
    add_prompt_col_sql = (
        f"ALTER TABLE heavyiq_nl_answers_v{version} ADD COLUMN prompt TEXT;"
    )
    add_answer_col_sql = (
        f"ALTER TABLE heavyiq_nl_answers_v{version} ADD COLUMN answer TEXT;"
    )
    update_prompt_col_sql = (
        f"UPDATE heavyiq_nl_answers_v{version} SET prompt = instruction;"
    )
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


def load_questions(
    con,
    questions_path,
    questions_instruction_prompt,
    questions_output_prompt,
    max_token_length,
    version,
    is_col_level,
    output_dir,
):
    questions_type = "column" if is_col_level else "table"
    drop_table_sql = (
        f"DROP TABLE IF EXISTS heavyiq_questions_{questions_type}_v{version}"
    )
    create_table_sql = f"CREATE TABLE heavyiq_questions_{questions_type}_v{version} (query_id INT, db_id TEXT, data_split TEXT, instruction TEXT, output TEXT, num_instruction_tokens INT);"
    load_sql = f"COPY heavyiq_questions_{questions_type}_v{version} FROM '{questions_path}' WITH (header='t');"
    add_prompt_col_sql = f"ALTER TABLE heavyiq_questions_{questions_type}_v{version} ADD COLUMN prompt TEXT;"
    add_answer_col_sql = f"ALTER TABLE heavyiq_questions_{questions_type}_v{version} ADD COLUMN answer TEXT;"
    update_prompt_col_sql = f"UPDATE heavyiq_questions_{questions_type}_v{version} SET prompt = instruction;"
    if questions_instruction_prompt and questions_output_prompt is not None:
        update_prompt_col_sql = f"UPDATE heavyiq_questions_{questions_type}_v{version} SET prompt = '{questions_instruction_prompt}\n' || instruction || '\n{questions_output_prompt}\n';"
    update_answer_col_sql = (
        f"UPDATE heavyiq_questions_{questions_type}_v{version} SET answer = output;"
    )
    export_train_sql = f"COPY (SELECT query_id, db_id, prompt, answer FROM heavyiq_questions_{questions_type}_v{version} WHERE data_split <> 'dev' AND num_instruction_tokens <= {max_token_length}) TO '{output_dir}/heavyiq_questions_{questions_type}_v{version}_train.csv' WITH (header='t');"
    export_eval_sql = f"COPY (SELECT query_id, db_id, prompt, answer FROM heavyiq_questions_{questions_type}_v{version} WHERE data_split = 'dev' AND num_instruction_tokens <= {max_token_length}) TO '{output_dir}/heavyiq_questions_{questions_type}_v{version}_eval.csv' WITH (header='t');"

    con.execute(drop_table_sql)
    con.execute(create_table_sql)
    con.execute(load_sql)
    con.execute(add_prompt_col_sql)
    con.execute(add_answer_col_sql)
    con.execute(update_prompt_col_sql)
    con.execute(update_answer_col_sql)
    con.execute(export_train_sql)
    con.execute(export_eval_sql)


def load_used_tables(
    con,
    used_tables_path,
    used_tables_instruction_prompt,
    used_tables_output_prompt,
    max_token_length,
    version,
    output_dir,
):
    table_suffix = f"v{version}_{max_token_length}_tokens"
    drop_table_sql = f"DROP TABLE IF EXISTS heavyiq_used_tables_{table_suffix}"
    create_table_sql = f"CREATE TABLE heavyiq_used_tables_{table_suffix} (query_id INT, db_id TEXT, data_split TEXT, instruction TEXT, output TEXT, num_instruction_tokens INT);"
    load_sql = f"COPY heavyiq_used_tables_{table_suffix} FROM '{used_tables_path}' WITH (header='t');"
    add_prompt_col_sql = (
        f"ALTER TABLE heavyiq_used_tables_{table_suffix} ADD COLUMN prompt TEXT;"
    )
    add_answer_col_sql = (
        f"ALTER TABLE heavyiq_used_tables_{table_suffix} ADD COLUMN answer TEXT;"
    )
    update_prompt_col_sql = (
        f"UPDATE heavyiq_used_tables_{table_suffix} SET prompt = instruction;"
    )
    if used_tables_instruction_prompt and used_tables_output_prompt is not None:
        update_prompt_col_sql = f"UPDATE heavyiq_used_tables_{table_suffix} SET prompt = '{used_tables_instruction_prompt}\n' || instruction || '\n{used_tables_output_prompt}\n';"
    update_answer_col_sql = (
        f"UPDATE heavyiq_used_tables_{table_suffix} SET answer = output;"
    )
    # Account for prompt tokens in max length
    export_train_sql = f"COPY (SELECT query_id, db_id, prompt, answer FROM heavyiq_used_tables_{table_suffix} WHERE data_split <> 'dev' AND num_instruction_tokens <= {max_token_length}) TO '{output_dir}/heavyiq_used_tables_{table_suffix}_train.csv' WITH (header='t');"
    export_eval_sql = f"COPY (SELECT query_id, db_id, prompt, answer FROM heavyiq_used_tables_{table_suffix} WHERE data_split = 'dev' AND num_instruction_tokens <= {max_token_length}) TO '{output_dir}/heavyiq_used_tables_{table_suffix}_eval.csv' WITH (header='t');"

    con.execute(drop_table_sql)
    con.execute(create_table_sql)
    con.execute(load_sql)
    con.execute(add_prompt_col_sql)
    con.execute(add_answer_col_sql)
    con.execute(update_prompt_col_sql)
    con.execute(update_answer_col_sql)
    con.execute(export_train_sql)
    con.execute(export_eval_sql)


def load_errors(
    con,
    errors_path,
    errors_instruction_prompt,
    errors_output_prompt,
    version,
    output_dir,
):
    drop_table_sql = f"DROP TABLE IF EXISTS heavyiq_errors_v{version}"
    create_table_sql = f"CREATE TABLE heavyiq_errors_v{version} (query_id INT, db_id TEXT, tables TEXT[], data_split TEXT, instruction TEXT, output TEXT);"
    load_sql = f"COPY heavyiq_errors_v{version} FROM '{errors_path}' WITH (header='t', array_marker='[]');"
    add_prompt_col_sql = (
        f"ALTER TABLE heavyiq_errors_v{version} ADD COLUMN prompt TEXT;"
    )
    add_answer_col_sql = (
        f"ALTER TABLE heavyiq_errors_v{version} ADD COLUMN answer TEXT;"
    )
    update_prompt_col_sql = (
        f"UPDATE heavyiq_errors_v{version} SET prompt = instruction;"
    )
    if errors_instruction_prompt and errors_output_prompt is not None:
        update_prompt_col_sql = f"UPDATE heavyiq_errors_v{version} SET prompt = '{errors_instruction_prompt}\n' || instruction || '\n{errors_output_prompt}\n';"
    update_answer_col_sql = f"UPDATE heavyiq_errors_v{version} SET answer = output;"
    export_train_sql = f"COPY (SELECT query_id, db_id, prompt, answer FROM heavyiq_errors_v{version} WHERE SAMPLE_RATIO(0.9)) TO '{output_dir}/heavyiq_errors_v{version}_train.csv' WITH (header='t');"
    export_eval_sql = f"COPY (SELECT query_id, db_id, prompt, answer FROM heavyiq_errors_v{version} WHERE NOT SAMPLE_RATIO(0.9)) TO '{output_dir}/heavyiq_errors_v{version}_eval.csv' WITH (header='t');"

    con.execute(drop_table_sql)
    con.execute(create_table_sql)
    con.execute(load_sql)
    con.execute(add_prompt_col_sql)
    con.execute(add_answer_col_sql)
    con.execute(update_prompt_col_sql)
    con.execute(update_answer_col_sql)
    con.execute(export_train_sql)
    con.execute(export_eval_sql)


def create_combo_dataset(
    con,
    version,
    max_token_length,
    combo_label,
    instruction_table,
    sql_table,
    nl_answers_table,
    questions_table_table,
    questions_column_table,
    used_tables_table,
    errors_table,
    only_sql_in_eval,
    output_dir,
):
    assert only_sql_in_eval, "only_sql_in_eval must be True"

    full_combo_label = (
        f"v{version}_{max_token_length}_tokens_{combo_label}"
        if combo_label is not None
        else f"v{version}_{max_token_length}_tokens"
    )

    drop_table_sql = f"DROP TABLE IF EXISTS heavyiq_combo_{full_combo_label}"
    create_table_sql = f"CREATE TABLE heavyiq_combo_{full_combo_label} (id INT, db_id TEXT, data_split TEXT, prompt TEXT, answer TEXT);"
    load_dolly_sql = f"INSERT INTO heavyiq_combo_{full_combo_label} SELECT id + 2000000, 'INVALID', 'train', prompt, answer FROM {instruction_table} WHERE length(prompt) < 1760 AND length(answer) < 640;"
    load_sql_sql = f"INSERT INTO heavyiq_combo_{full_combo_label} SELECT query_id, db_id, CASE WHEN data_split <> 'dev' THEN 'train' ELSE 'eval' END, prompt, answer FROM {sql_table} WHERE num_targeted_instruction_tokens < {max_token_length};"
    load_answers_sql = None
    if nl_answers_table is not None:
        load_answers_sql = f"INSERT INTO heavyiq_combo_{full_combo_label} SELECT query_id + 1000000, db_id, 'train', prompt, answer FROM {nl_answers_table} WHERE data_split <> 'dev';"
    load_questions_table_sql = None
    if questions_table_table is not None:
        load_questions_table_sql = f"INSERT INTO heavyiq_combo_{full_combo_label} SELECT query_id + 3000000, db_id, 'train', prompt, answer FROM {questions_table_table} WHERE data_split <> 'dev' AND num_instruction_tokens < {max_token_length};"
    load_questions_column_sql = None
    if questions_column_table is not None:
        load_questions_column_sql = f"INSERT INTO heavyiq_combo_{full_combo_label} SELECT query_id + 4000000, db_id, 'train', prompt, answer FROM {questions_column_table} WHERE data_split <> 'dev' AND num_instruction_tokens < {max_token_length};"
    load_used_tables_table_sql = None
    if used_tables_table is not None:
        load_used_tables_table_sql = f"INSERT INTO heavyiq_combo_{full_combo_label} SELECT query_id + 5000000, db_id, 'train', prompt, answer FROM {used_tables_table} WHERE data_split <> 'dev' AND num_instruction_tokens < {max_token_length};"
    load_errors_sql = None
    if errors_table is not None:
        load_errors_sql = f"INSERT INTO heavyiq_combo_{full_combo_label} SELECT query_id + 6000000, db_id, 'train', prompt, answer FROM {errors_table};"
    export_train_sql = f"COPY (SELECT id, db_id, prompt, answer FROM heavyiq_combo_{full_combo_label} WHERE data_split = 'train') TO '{output_dir}/heavyiq_combo_{full_combo_label}_train.csv' WITH (header='t');"
    export_eval_sql = f"COPY (SELECT id, db_id, prompt, answer FROM heavyiq_combo_{full_combo_label} WHERE data_split = 'eval') TO '{output_dir}/heavyiq_combo_{full_combo_label}_eval.csv' WITH (header='t');"

    con.execute(drop_table_sql)
    con.execute(create_table_sql)
    con.execute(load_dolly_sql)
    con.execute(load_sql_sql)
    if load_answers_sql is not None:
        print(load_answers_sql)
        con.execute(load_answers_sql)
    if load_questions_table_sql is not None:
        con.execute(load_questions_table_sql)
    if load_questions_column_sql is not None:
        con.execute(load_questions_column_sql)
    if load_used_tables_table_sql is not None:
        con.execute(load_used_tables_table_sql)
    if load_errors_sql is not None:
        con.execute(load_errors_sql)
    con.execute(export_train_sql)
    con.execute(export_eval_sql)


def main(argv):
    options = getOptions(argv)
    con = heavyai.connect(
        user=options.user,
        password=options.password,
        host=options.host,
        dbname=options.database,
    )
    queries_table = None
    nl_answers_table = None
    questions_table_table = None
    questions_column_table = None
    used_tables_table = None
    errors_table = None
    if options.queries is not None:
        queries_table = load_queries(
            con,
            options.queries,
            options.query_instruction_prompt,
            options.query_output_prompt,
            options.max_token_length,
            options.label,
            options.version,
            options.output_dir,
            options.num_query_shards,
            options.min_avg_prob,
            options.avg_prob_order_noise,
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
        nl_answers_table = f"heavyiq_nl_answers_v{options.version}"
    if options.questions_table_level is not None:
        load_questions(
            con,
            options.questions_table_level,
            options.questions_table_level_instruction_prompt,
            options.questions_table_level_output_prompt,
            options.max_token_length,
            options.version,
            False,
            options.output_dir,
        )
        questions_table_table = f"heavyiq_questions_table_v{options.version}"

    if options.questions_column_level is not None:
        load_questions(
            con,
            options.questions_column_level,
            options.questions_column_level_instruction_prompt,
            options.questions_column_level_output_prompt,
            options.max_token_length,
            options.version,
            True,
            options.output_dir,
        )
        questions_column_table = f"heavyiq_questions_column_v{options.version}"

    if options.used_tables is not None:
        load_used_tables(
            con,
            options.used_tables,
            options.used_tables_instruction_prompt,
            options.used_tables_output_prompt,
            options.max_token_length,
            options.version,
            options.output_dir,
        )
        used_tables_table = (
            f"heavyiq_used_tables_v{options.version}_{options.max_token_length}_tokens"
        )

    if options.errors is not None:
        load_errors(
            con,
            options.errors,
            options.errors_instruction_prompt,
            options.errors_output_prompt,
            options.version,
            options.output_dir,
        )
        errors_table = f"heavyiq_errors_v{options.version}"

    if options.create_combo_dataset and options.queries and options.nl_answers:
        create_combo_dataset(
            con,
            options.version,
            options.max_token_length,
            options.label,
            options.instruction_table,
            queries_table,
            nl_answers_table,
            questions_table_table,
            questions_column_table,
            used_tables_table,
            errors_table,
            True,
            options.output_dir,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
