import argparse
import csv
import os
import sqlite3
import sys


def getOptions(argv=None):
    parser = argparse.ArgumentParser(description="Export Sqlite database")
    parser.add_argument("-d", "--database", help="Sqlite database")
    return parser.parse_args(argv)


def get_create_table_statement(cursor, table_name):
    cursor.execute(f"""PRAGMA table_info("{table_name}")""")
    columns = cursor.fetchall()

    column_statements = []
    for column in columns:
        col_name = column[1]
        if col_name[0] != '"':
            col_name = '"' + column[1] + '"'
        col_type = column[2]
        if col_type == "DATETIME":
            col_type = "TIMESTAMP"
        elif col_type == "BLOB":
            col_type = "TEXT ENCODING NONE"
        col_statement = f"{col_name} {col_type}"
        column_statements.append(col_statement)

    create_table_statement = (
        f"""CREATE TABLE "{table_name}" ({', '.join(column_statements)});"""
    )
    return create_table_statement


def export_table_to_csv(cursor, table_name, csv_file_path):
    cursor.execute(f"""SELECT * FROM "{table_name}" """)
    rows = cursor.fetchall()

    with open(csv_file_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        # Write the header
        cursor.execute(f"""PRAGMA table_info("{table_name}")""")
        columns = [column[1] for column in cursor.fetchall()]
        writer.writerow(columns)

        # Write the data
        for row in rows:
            writer.writerow(row)


def main(argv):
    options = getOptions(argv)
    # Connect to SQLite database
    print(options.database)
    db_name = options.database.split("/")[-1].split(".")[0]
    print(db_name)
    conn = sqlite3.connect(options.database)
    cursor = conn.cursor()

    # Fetch all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = [table[0] for table in cursor.fetchall()]

    # Create a directory to hold the output files
    # db_name = options.database.split(".")[0].split("/")[-1]
    output_dir = f"data_exports/bird/{db_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Export each table
    for table_name in table_names:
        print(f"Exporting {table_name}...")

        # Get CREATE TABLE statement and save to a file
        create_statement = get_create_table_statement(cursor, table_name)
        with open(f"{output_dir}/{table_name}_create.sql", "w") as f:
            f.write(create_statement)

        # Export table data to CSV
        export_table_to_csv(cursor, table_name, f"{output_dir}/{table_name}.csv")

    # Close the connection
    conn.close()


if __name__ == "__main__":
    main(sys.argv[1:])
