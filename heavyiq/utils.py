import re
from multiprocessing.managers import SyncManager
from typing import Generic, TypeVar, Optional


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


KT = TypeVar("KT")  # Key type
VT = TypeVar("VT")  # Value type


class LRUCache(Generic[KT, VT]):
    """
    Implements Singleton/shared caching ie. shared cache which can be
    accessed by any process.
    """

    def __init__(self, capacity: int = 100, manager: SyncManager | None = None) -> None:
        self.capacity: int = capacity
        if manager:
            # this should create seperate manager processes if manager isn't passed
            # thread safe/process-safe shared dict which holds the key, value pair
            self.cache = manager.dict()
            # shared list which holds the key order, ie. recently used key should be removed and appended to the last
            self.order = manager.list()
        else:
            self.cache = {}  # type: ignore
            self.order = []  # type: ignore

    def get(self, key: KT) -> Optional[VT]:
        """
        Returns the value of passed dict key if exists else return None.
        """
        if key in self.cache:
            # Move the key to the end (most recently used) in the order list
            self.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: KT, value: VT) -> None:
        """
        Helps to put the given key, value pair on the manager.Dict.
        """
        if key in self.cache:
            # If the key already exists, update its value and move it to the end
            self.cache[key] = value
            self.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                # If cache is full, evict the least recently used item
                self.evict_lru()
            # Add the new key-value pair
            self.cache[key] = value
            # Add the key to the end (most recently used) in the order list
            self.order.append(key)

    def move_to_end(self, key: KT):
        """
        Move the key to the end (most recently used) in the order list.

        Args:
            key (str): key to be moved.
        """
        self.order.remove(key)
        self.order.append(key)

    def evict_lru(self):
        """
        Helps to remove the recently used key from cache and updates the order list accordingly.
        This just makes space for the new item when threashold limit reached.
        """
        # Get the least recently used key from the front of the order list
        lru_key = self.order[0]
        # Remove the least recently used key from the cache
        del self.cache[lru_key]
        # Remove the least recently used key from the front of the order list
        self.order.pop(0)
