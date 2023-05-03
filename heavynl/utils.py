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
    return re.sub(r"^\n+|\n+$", "", sql).strip()


def is_destructive_sql(sql: str) -> bool:
    """
    Determines if a provided SQL statement is destructive or causes a modification.

    Args:
        sql (str): The SQL statement to be checked.

    Returns:
        bool: True if the SQL statement is destructive, False if it is non-destructive.
    """
    # Check for destructive SQL statements
    destructive_statements = {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "REPLACE", "CREATE"}

    # Convert the SQL statement to uppercase and split it by whitespace
    sql_parts = sql.strip().upper().split()

    # Check if the first part of the SQL statement is in the destructive_statements set
    return sql_parts[0] in destructive_statements
