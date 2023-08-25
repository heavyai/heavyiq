import argparse
import copy
import heavyai
import pandas as pd
import re
import sys
from typing import Dict, Tuple


def getOptions(argv=None):
    parser = argparse.ArgumentParser(description="Text HeavySQL literal extraction.")
    parser.add_argument("-s", "--host", help="HeavyDB server address", default="localhost")
    parser.add_argument("-p", "--port", help="HeavyDB server port", default="6273")
    parser.add_argument("-u", "--user", help="HeavyDB user name", default="admin")
    parser.add_argument("-w", "--password", help="HeavyDB password", default="HyperInteractive")
    parser.add_argument("-d", "--db", help="HeavyDB DB", default="heavyai")
    parser.add_argument("-q", "--query", help="SQL query", default=None)
    return parser.parse_args(argv)


def get_query_plan(con, query):
    explain_calcite_query = f"EXPLAIN CALCITE DETAILED {query}"
    res = con.execute(explain_calcite_query)
    query_plan = list(res)[0][0]
    return query_plan


def extract_column_mappings(query_plan: str) -> Dict[str, Tuple[str, str, str]]:
    result = {}
    # Use a regex pattern to capture (database, table, column) and literal values in LogicalFilter
    # pattern = r"\[\$([0-9]+)->(db:[\w]+),tableName:([\w]+),colName:([\w]+)\]\], \=.*'([^']+)')"
    # pattern = r"\[\$([0-9]+)->(db:[\w]+),tableName:([\w]+),colName:([\w]+)\], \=.*'([^']+)'\)"
    pattern = r"\[\$([0-9]+)->db:([\w]+),tableName:([\w]+),colName:([\w]+)\]"

    matches = re.findall(pattern, query_plan)

    # for _, db, table, column, literal in matches:
    for id, db, table, column in matches:
        col_id = (db, table, column)
        result[id] = col_id

    return result


def extract_str_literal_ops(query_plan):
    result = {}
    pattern1 = r"(LIKE|PG_ILIKE|>=|<=|<>|=)\(\$(\d+), '([\w ]+)'"
    pattern2 = r"(LIKE|PG_ILIKE|>=|<=|<>|=)\('([\w+ ]+)', \$(\d+)\)"

    matches1 = re.findall(pattern1, query_plan)
    matches2 = re.findall(pattern2, query_plan)
    for op, id, literal in matches1:
        result[id] = (op, literal)
    for op, literal, id in matches2:
        result[id] = (op, literal)
    return result


def join_str_literals_with_col_mapping(str_literal_ops, col_mapping):
    result = []
    for id, (op, literal) in str_literal_ops.items():
        db, table, column = col_mapping[id]
        result.append({"operator": op, "literal": literal, "database": db, "table": table, "column": column})
    return result


def matchLiteral(con, literal, exact_match_threshold):
    case_match_query = f"SELECT {literal['column']}, COUNT(*) FROM {literal['database']}.{literal['table']} WHERE {literal['column']} ILIKE '{literal['literal']}' GROUP BY {literal['column']} ORDER BY COUNT(*) DESC;"
    print(case_match_query)
    case_match_df = pd.read_sql(case_match_query, con)
    print(case_match_df)
    num_case_match_rows = len(case_match_df.axes[0])
    num_case_match_cols = len(case_match_df.axes[1])
    print(f"Rows: {num_case_match_rows}, Columns: {num_case_match_cols}")
    case_match_df.reset_index()
    exact_match_count = 0
    total_count = 0
    altered_literal = copy.deepcopy(literal)
    for index, row in case_match_df.iterrows():
        print(f"{row[0]}: {row[1]}")
        if row[0] == literal["literal"]:
            exact_match_count += row[1]
        total_count += row[1]
    print(total_count)
    if total_count > 0 and literal["operator"] != "ILIKE":
        if exact_match_count == 0 and num_case_match_rows == 1:
            altered_literal["literal"] = case_match_df.iloc[0, 0]
            print(f"Changing case of literal from {literal['literal']} to {altered_literal['literal']}")
            return altered_literal
        elif exact_match_count / total_count < exact_match_threshold:
            print(f"Found {exact_match_count} exact matches out of {total_count} total matches")
            if literal["operator"] == "<>":
                altered_literal["operator"] = "NOT ILIKE"
            else:
                altered_literal["operator"] = "ILIKE"
            return altered_literal
    elif total_count == 0:
        lower_literal = literal["literal"].lower()
        # similarity_query = f"SELECT LOWER({literal['column']}), COUNT(*), JAROWINKLER_SIMILARITY(LOWER({literal['column']}), '{lower_literal}') FROM {literal['database']}.{literal['table']} WHERE JAROWINKLER_SIMILARITY(LOWER({literal['column']}), '{lower_literal}') >= 80 GROUP BY LOWER({literal['column']}) ORDER BY JAROWINKLER_SIMILARITY(LOWER({literal['column']}), '{lower_literal}') DESC NULLS LAST LIMIT 1;"
        similarity_query = f"WITH distinct_values AS (SELECT LOWER({literal['column']}) AS lower_attr, COUNT(*) AS n FROM {literal['database']}.{literal['table']} GROUP BY LOWER({literal['column']})) SELECT lower_attr, LEVENSHTEIN_DISTANCE(lower_attr, '{lower_literal}') - ABS(LENGTH(lower_attr) - LENGTH('{lower_literal}')) FROM distinct_values WHERE LEVENSHTEIN_DISTANCE(lower_attr, '{lower_literal}') - ABS(LENGTH(lower_attr) - LENGTH('{lower_literal}')) < 5 ORDER BY LEVENSHTEIN_DISTANCE(lower_attr, '{lower_literal}') - ABS(LENGTH(lower_attr) - LENGTH('{lower_literal}')) ASC LIMIT 1;"
        # LOWER({literal['column']}), COUNT(*), JAROWINKLER_SIMILARITY(LOWER({literal['column']}), '{lower_literal}') FROM {literal['database']}.{literal['table']} WHERE JAROWINKLER_SIMILARITY(LOWER({literal['column']}), '{lower_literal}') >= 80 GROUP BY LOWER({literal['column']}) ORDER BY JAROWINKLER_SIMILARITY(LOWER({literal['column']}), '{lower_literal}') DESC NULLS LAST LIMIT 1;"
        print(similarity_query)
        similarity_df = pd.read_sql(similarity_query, con)
        num_similarity_rows = len(similarity_df.axes[0])
        print(similarity_df)
        if num_similarity_rows > 0:
            altered_literal["literal"] = similarity_df.iloc[0, 0]
            print(f"Changing literal from {literal['literal']} to {altered_literal['literal']}")
            if literal["operator"] == "<>":
                altered_literal["operator"] = "NOT ILIKE"
            else:
                altered_literal["operator"] = "ILIKE"
        return altered_literal
    return altered_literal


def main(argv):
    options = getOptions(argv)
    con = heavyai.connect(user=options.user, password=options.password, host=options.host, dbname=options.db)
    query_plan = get_query_plan(con, options.query)
    print(query_plan)
    col_mapping = extract_column_mappings(query_plan)
    print(col_mapping)
    str_literal_ops = extract_str_literal_ops(query_plan)
    print(str_literal_ops)
    str_literal_ops_with_col_mapping = join_str_literals_with_col_mapping(str_literal_ops, col_mapping)
    print(str_literal_ops_with_col_mapping)
    exact_match_threshold = 0.999999
    altered_query = copy.deepcopy(options.query)
    for str_literal_op in str_literal_ops_with_col_mapping:
        print(str_literal_op)
        altered_str_literal_op = matchLiteral(con, str_literal_op, exact_match_threshold)
        if altered_str_literal_op != str_literal_op:
            if altered_str_literal_op["operator"] != str_literal_op["operator"]:
                altered_query = altered_query.replace(
                    f"{str_literal_op['column']} {str_literal_op['operator']} '{str_literal_op['literal']}'",
                    f"{str_literal_op['column']} {altered_str_literal_op['operator']} '{altered_str_literal_op['literal']}'",
                )
                print(f"Operator changed from {str_literal_op['operator']} to {altered_str_literal_op['operator']}")

            if altered_str_literal_op["literal"] != str_literal_op["literal"]:
                altered_query = altered_query.replace(
                    f"{str_literal_op['column']} {altered_str_literal_op['operator']} '{str_literal_op['literal']}'",
                    f"{str_literal_op['column']} {altered_str_literal_op['operator']} '{altered_str_literal_op['literal']}'",
                )
                print(f"Literal changed from {str_literal_op['literal']} to {altered_str_literal_op['literal']}")
    print(altered_query)
    final_df = pd.read_sql(altered_query, con)
    print(final_df)


if __name__ == "__main__":
    main(sys.argv[1:])
