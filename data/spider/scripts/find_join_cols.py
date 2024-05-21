import argparse
import heavyai
import sys


def getOptions(argv=None):
    parser = argparse.ArgumentParser(description="Find heavyai join columns")
    parser.add_argument("-s", "--host", help="HeavyDB server address", default="10.211.55.4")
    parser.add_argument("-p", "--port", help="HeavyDB server port", default="6273")
    parser.add_argument("-u", "--user", help="HeavyDB user name", default="admin")
    parser.add_argument("-w", "--password", help="HeavyDB password", default="HyperInteractive")
    parser.add_argument("-d", "--database", help="HeavyDB database", default="heavyai")
    parser.add_argument("--table-1", help="Table 1", required=True)
    parser.add_argument("--table-2", help="Table 2", required=True)

    return parser.parse_args(argv)


def getTableColumns(con, table):
    table_details = con.get_table_details(table)
    table_cols = [{"name": col[0], "type": col[1], "encoding": col[6]} for col in table_details]
    return table_cols


def getLargerSmallerTable(con, table_1, table_2):
    table_1_rows = list(con.execute(f"""SELECT COUNT(*) FROM "{table_1}" """))[0][0]
    table_2_rows = list(con.execute(f"""SELECT COUNT(*) FROM "{table_2}" """))[0][0]
    if table_1_rows >= table_2_rows:
        return (
            table_1,
            table_2,
        )
    else:
        return table_2, table_1


def getTableCardinality(con, table):
    return list(con.execute(f"""SELECT COUNT(*) FROM "{table}" """))[0][0]


def getColumnCardinality(con, table, column):
    return list(con.execute(f"""SELECT COUNT(DISTINCT "{column}") FROM "{table}" """))[0][0]


def main(argv):
    options = getOptions(argv)
    con = heavyai.connect(user=options.user, password=options.password, host=options.host, dbname=options.database)
    db_tables = [table.lower() for table in con.get_tables()]
    table_1 = options.table_1.lower()
    table_2 = options.table_2.lower()
    if table_1 not in db_tables:
        print(f"Table {table_1} not found in database {options.database}")
        sys.exit(1)
    if table_2 not in db_tables:
        print(f"Table {table_2} not found in database {options.database}")
        sys.exit(1)
    larger_table, smaller_table = getLargerSmallerTable(con, table_1, table_2)
    print(f"Larger table: {larger_table}")
    larger_table_cols = getTableColumns(con, larger_table)
    smaller_table_cols = getTableColumns(con, smaller_table)

    larger_str_cols = [col for col in larger_table_cols if col["type"] == "STR" and col["encoding"] == "DICT"]
    smaller_table_str_cols = [col for col in smaller_table_cols if col["type"] == "STR" and col["encoding"] == "DICT"]

    join_frac_threshold = 0.7

    for larger_table_str_col in larger_str_cols:
        larger_col_cardinality = getColumnCardinality(con, larger_table, larger_table_str_col["name"])
        if larger_col_cardinality <= 1:
            continue
        for smaller_table_str_col in smaller_table_str_cols:
            join_sql = f"""SELECT COUNT(*) FROM (SELECT DISTINCT "{larger_table_str_col['name']}" AS uniq_vals FROM "{larger_table}") a JOIN (SELECT DISTINCT "{smaller_table_str_col['name']}" AS uniq_vals FROM "{smaller_table}") b ON a.uniq_vals = b.uniq_vals"""
            joined_num_rows = list(con.execute(join_sql))[0][0]
            join_frac = joined_num_rows * 1.0 / larger_col_cardinality
            if join_frac > join_frac_threshold:
                print(
                    f"Join columns found: {larger_table}.{larger_table_str_col['name']} = {smaller_table}.{smaller_table_str_col['name']} with join fraction {join_frac}"
                )


if __name__ == "__main__":
    main(sys.argv[1:])
