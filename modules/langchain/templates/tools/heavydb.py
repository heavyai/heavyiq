QueryHeavyDBSchemaToolDescription = """
Use this tool to retrieve the schemas for a query.

The input is a prompt that will be handled by a LLM with access to a HeavyDB index that contains schemas, sample rows, and summaries of each table in the database.
Prepare a prompt that the LLM can use to fetch the information you require to complete your objective.
The output will be schemas for the tables that the LLM thinks are relevant to the prompt.
"""

QueryHeavyDBToolDescription = """
ONLY USE WITH KNOWN TABLE SCHEMAS. This tool is for querying HeavyDB databases.

Input an accurate SQL query for output. Apply query limits. If an error occurs, revise and retry. Be familiar with table schemas and avoid common mistakes, such as:

- NULL values with NOT IN
- Using UNION instead of UNION ALL
- BETWEEN for exclusive ranges
- Data type mismatch
- Incorrectly quoting identifiers
- Wrong number of arguments in functions
- Incorrect data type casting
- Improper join columns
- Reserved SQL keywords as aliases
"""

InfoHeavyDBToolDescription = """
Provide a comma-separated list of tables as input, and this tool will output the schema and sample rows for those tables.

Example Input: table1, table2, table3
Refrain from including single quotes in the table names.
"""
