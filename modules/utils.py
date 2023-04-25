import re


def strip_sql_comments(sql: str) -> str:
    # Remove block comments
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    # Remove single-line comments
    sql = re.sub(r"--.*$", "", sql, flags=re.MULTILINE)
    # Remove triple backticks
    sql = re.sub(r"```", "", sql)
    # Remove leading and trailing double-quotes and single quotes
    sql = sql.strip("'\"")
    # Remove leading and trailing newlines
    return re.sub(r"^\n+|\n+$", "", sql)
