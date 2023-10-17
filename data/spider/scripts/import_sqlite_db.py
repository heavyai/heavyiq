import argparse
import heavyai
import os
import sys


def getOptions(argv=None):
    parser = argparse.ArgumentParser(description="Restore dumped sqlite database")
    parser.add_argument("-s", "--host", help="HeavyDB server address", default="localhost")
    parser.add_argument("-p", "--port", help="HeavyDB server port", default="6273")
    parser.add_argument("-u", "--user", help="HeavyDB user name", default="admin")
    parser.add_argument("-w", "--password", help="HeavyDB password", default="HyperInteractive")
    parser.add_argument("-d", "--directory", help="Data directory", default=None)
    parser.add_argument(
        "--prefix", help="Directory prefix", default="/media/psf/Home/projects/heavynl/data/spider/scripts"
    )
    return parser.parse_args(argv)


def main(argv):
    options = getOptions(argv)
    db_name = options.directory.split("/")[-1]
    heavyai_con = heavyai.connect(user=options.user, password=options.password, host=options.host, dbname="heavyai")
    heavyai_con.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")

    db_con = heavyai.connect(user=options.user, password=options.password, host=options.host, dbname=db_name)

    for file in os.listdir(options.directory):
        if file.endswith(".sql"):
            filepath = os.path.join(options.directory, file)
            with open(filepath, "r") as file:
                sql_content = file.read()
            try:
                db_con.execute(sql_content)
            except Exception as e:
                print(f"SQL error occurred while executing {filepath}: {e}")

    for file in os.listdir(options.directory):
        if file.endswith(".csv"):
            filepath = os.path.join(options.directory, file)
            full_path = options.prefix + "/" + filepath
            table_name = file.split(".")[0]
            import_sql = f"""COPY "{table_name}" FROM '{full_path}' WITH (HEADER='t')"""
            try:
                db_con.execute(import_sql)
            except Exception as e:
                print(f"SQL error occurred while executing {import_sql}: {e}")


if __name__ == "__main__":
    main(sys.argv[1:])
