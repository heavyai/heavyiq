import re


def strip_sql_comments(sql: str) -> str:
    # Remove everything before a block comment
    sql = re.sub(r"^.*?/\*", "/*", sql, flags=re.DOTALL)
    # Remove block comments
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    # Remove single-line comments
    sql = re.sub(r"--.*$", "", sql, flags=re.MULTILINE)
    # Remove anything after the last semicolon
    sql = re.sub(r";[^;]*$", ";", sql, flags=re.DOTALL)
    # Remove triple backticks, and ```sql if present
    sql = re.sub(r"```(sql)?", "", sql)
    # Remove leading and trailing double-quotes and single quotes
    sql = sql.strip("'\"")
    # Remove leading and trailing newlines
    sql = re.sub(r"^\n+|\n+$", "", sql).strip()
    # If SQL doesn't end with a semicolon, add one
    if not sql.endswith(";"):
        sql += ";"
    return sql


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


def rate_sql_complexity(plan: str) -> int:
    """
    Rates the complexity of a SQL query based on the provided plan string.
    """

    # define patterns for each level of complexity
    patterns = {
        5: r"LogicalJoin|LogicalCorrelate|LogicalUnion|RelLeftDeepInnerJoin|RexWindowFunctionOperator",
        4: r"LogicalProject|RexSubQuery",
        3: r"LogicalAggregate|RexAgg",
        2: r"LogicalFilter|RexLiteral|RexOperator",
    }

    # start with the lowest complexity
    complexity = 1

    # check for features indicating higher complexity
    for level in range(5, 1, -1):
        if re.search(patterns[level], plan, re.IGNORECASE):
            complexity = level
            break

    # return the final complexity rating
    return complexity
