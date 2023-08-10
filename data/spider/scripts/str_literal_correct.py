import argparse
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


def matchLiteral(con, literal):
    case_match_query = f"SELECT {literal['column']}, COUNT(*) FROM {literal['database']}.{literal['table']} WHERE {literal['column']} ILIKE {literal['literal']} GROUP BY {literal['column']} ORDER BY COUNT(*) DESC;"
    print(case_match_query)
    df = pd.read_sql(case_match_query, con)
    print(df)
    num_rows = len(df.axes[0])
    num_cols = len(df.axes[1])
    print(f"Rows: {num_rows}, Columns: {num_cols}")


def getLiterals(con, query):
    literals = []
    explain_calcite_query = f"EXPLAIN CALCITE {query}"
    print(explain_calcite_query)
    res = con.execute(explain_calcite_query)
    query_plan = list(res)[0][0]
    # Extract lines that have literals
    literal_lines = re.findall(
        r"(LogicalFilter\(condition=\[)(LIKE|ILIKE|>=|<=|<>|=)(.*?'\w+'.*?{\[\$.*?\]})", query_plan
    )

    literal_results = []
    for line in literal_lines:
        # Extract the operator
        operator = line[1]

        # Extract the literal
        literal = re.search(r"'(\w+)'", line[2]).group(1)

        mappings = re.search(r"{\[\$(\d+)->db:(\w+),tableName:(\w+),colName:(\w+)\]}", line[2]).groups()

        literal_result = {
            "operator": operator,
            "literal": f"'{literal}'",
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
    for literal in literals:
        matchLiteral(con, literal)


if __name__ == "__main__":
    main(sys.argv[1:])
