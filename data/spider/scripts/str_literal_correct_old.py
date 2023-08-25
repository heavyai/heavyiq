import argparse
import copy
import heavyai
import pandas as pd
import re
import sys


def getOptions(argv=None):
    parser = argparse.ArgumentParser(description="Text HeavySQL literal extraction.")
    parser.add_argument("-s", "--host", help="HeavyDB server address", default="localhost")
    parser.add_argument("-p", "--port", help="HeavyDB server port", default="6273")
    parser.add_argument("-u", "--user", help="HeavyDB user name", default="admin")
    parser.add_argument("-w", "--password", help="HeavyDB password", default="HyperInteractive")
    parser.add_argument("-d", "--db", help="HeavyDB DB", default="heavyai")
    parser.add_argument("-q", "--query", help="SQL query", default=None)
    return parser.parse_args(argv)


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
        similarity_query = f"SELECT LOWER({literal['column']}), COUNT(*), JAROWINKLER_SIMILARITY(LOWER({literal['column']}), '{lower_literal}') FROM {literal['database']}.{literal['table']} WHERE JAROWINKLER_SIMILARITY(LOWER({literal['column']}), '{lower_literal}') >= 80 GROUP BY LOWER({literal['column']}) ORDER BY JAROWINKLER_SIMILARITY(LOWER({literal['column']}), '{lower_literal}') DESC NULLS LAST LIMIT 1;"
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


def getLiterals(con, query):
    literals = []
    explain_calcite_query = f"EXPLAIN CALCITE {query}"
    print(explain_calcite_query)
    res = con.execute(explain_calcite_query)
    query_plan = list(res)[0][0]
    # Extract lines that have literals
    literal_lines = re.findall(
        r"(LogicalFilter\(condition=\[)(LIKE|ILIKE|>=|<=|<>|=)(.*?'([^']+).*?{\[\$.*?\]})", query_plan
    )

    # r"(LogicalFilter\(condition=\[)(LIKE|ILIKE|>=|<=|<>|=)(.*?'\w+'.*?{\[\$.*?\]})", query_plan
    literal_results = []
    for line in literal_lines:
        # Extract the operator
        operator = line[1]

        # Extract the literal
        # literal = re.search(r"'(\w+)'", line[2]).group(1)
        literal = re.search(r"'([^']+)'", line[2]).group(1)

        mappings = re.search(r"{\[\$(\d+)->db:(\w+),tableName:(\w+),colName:(\w+)\]}", line[2]).groups()

        literal_result = {
            "operator": operator,
            "literal": f"{literal}",
            "database": mappings[1],
            "table": mappings[2],
            "column": mappings[3],
        }

        literal_results.append(literal_result)

    return literal_results


def main(argv):
    options = getOptions(argv)
    con = heavyai.connect(user=options.user, password=options.password, host=options.host, dbname=options.db)
    literals = getLiterals(con, options.query)
    print(literals)
    exact_match_threshold = 0.999999
    altered_query = copy.deepcopy(options.query)
    for literal in literals:
        altered_literal = matchLiteral(con, literal, exact_match_threshold)
        if altered_literal != literal:
            if altered_literal["operator"] != literal["operator"]:
                altered_query = altered_query.replace(
                    f"{literal['column']} {literal['operator']} '{literal['literal']}'",
                    f"{literal['column']} {altered_literal['operator']} '{literal['literal']}'",
                )
                print(f"Operator changed from {literal['operator']} to {altered_literal['operator']}")

            if altered_literal["literal"] != literal["literal"]:
                altered_query = altered_query.replace(
                    f"{literal['column']} {altered_literal['operator']} '{literal['literal']}'",
                    f"{literal['column']} {altered_literal['operator']} '{altered_literal['literal']}'",
                )
                print(f"Literal changed from {literal['literal']} to {altered_literal['literal']}")
    print(altered_query)
    final_df = pd.read_sql(altered_query, con)
    print(final_df)


if __name__ == "__main__":
    main(sys.argv[1:])
