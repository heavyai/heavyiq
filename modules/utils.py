import re


def strip_sql_comments(sql: str) -> str:
    # Remove block comments
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    # Remove single-line comments
    return re.sub(r"--.*$", "", sql, flags=re.MULTILINE).strip()
