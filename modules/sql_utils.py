"""
Responsible for generating SQL queries based on natural language instructions. This submodule should contain functions
that take natural language input, parse it, and generate corresponding SQL queries using the provided schema.
"""

from __future__ import annotations
import re
from typing import TYPE_CHECKING

import modules.heavydb as db
from modules.logging_utils import default_logger as logger

if TYPE_CHECKING:
    from heavyai import Cursor
    from modules.chat_interaction import ChatManager


MAX_SQL_TRY_COUNT = 3


def extract_sql_from_response(response: str) -> str:
    """
    Extracts a 'SELECT' statement from string, if present
    """
    stmt = response.replace("\n", " ").replace("\r", " ")
    pattern = r"(SELECT.+?;)"
    result = re.search(pattern, stmt, re.IGNORECASE)
    if result:
        return result.group(0)
    raise Exception("No statement of specified type found in provided string")


def verify_and_attempt_fix_sql(
    sql_statement: str, chat_manager: ChatManager, remaining_tries: int = MAX_SQL_TRY_COUNT
) -> tuple[str, Cursor]:
    """
    Verifies the provided SQL statement by executing it. If an exception occurs, attempts to fix it by
    prompting the AI model with the exception message and trying again. The function recursively
    retries until either the SQL statement is successfully executed or the maximum number of tries is reached.

    Args:
        sql_statement (str): The SQL statement to verify and potentially fix.
        chat_manager (ChatManager): The ChatManager instance for interacting with the AI model.
        remaining_tries (int, optional): The number of retries remaining. Defaults to MAX_SQL_TRY_COUNT.

    Returns:
        tuple[str, Cursor]: A tuple containing the final SQL statement and the corresponding Cursor object.

    Raises:
        Exception: If the maximum number of tries is reached and the SQL statement is still invalid.
    """
    try:
        return sql_statement, db.conn.execute(sql_statement)
    except Exception as error:
        logger.info(f"Invalid SQL: {sql_statement}")
        logger.debug(str(error))
        if remaining_tries <= 0:
            raise error
        logger.info("Trying again...")
        chat_manager.add_message("user", str(error))
        sql_statement = extract_sql_from_response(chat_manager.prompt_ai())
        return verify_and_attempt_fix_sql(sql_statement, chat_manager, remaining_tries=remaining_tries - 1)
