import asyncio
import re
from multiprocessing import Manager
from multiprocessing.managers import SyncManager
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

import aiofiles
from fastapi.concurrency import run_in_threadpool


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


def calc_query_stats(plan: str) -> dict[str, int]:
    sql_features = {
        "joins": ["LogicalJoin", "RelLeftDeepInnerJoin"],
        "unions": ["LogicalUnion"],
        "aggs": ["LogicalAggregate"],
        "filters": ["LogicalFilter"],
        "sorts": ["LogicalSort"],
    }

    query_stats = {}
    for sql_feature, search_patterns in sql_features.items():
        num_features_found = 0
        for search_pattern in search_patterns:
            num_features_found += plan.count(search_pattern)
        query_stats[sql_feature] = num_features_found
    return query_stats


KT = TypeVar("KT")  # Key type
VT = TypeVar("VT")  # Value type


class SharedDictSingleton(Generic[KT, VT]):
    _instance: "SharedDictSingleton[KT, VT]" = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            manager = Manager()
            cls._instance._manager = manager
            cls._instance._shared_dict = manager.dict()
        return cls._instance

    async def get(self, key: KT) -> Any:
        async with self._lock:
            return self._shared_dict.get(key)  # type: ignore

    async def put(self, key: KT, value: VT) -> None:
        async with self._lock:
            self._shared_dict[key] = value  # type: ignore

    def sget(self, key: KT) -> Any:  # sync get where manager.Dict().get and put are atomic, thus avoids race-conditions
        return self._shared_dict.get(key)  # type: ignore

    def sput(self, key: KT, value: VT) -> None:
        self._shared_dict[key] = value  # type: ignore


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

    def delete(self, key: KT) -> None:
        """
        Deletes the key and it's associated value from the cache dict.
        """
        try:
            del self.cache[key]
            self.order.remove(key)
        except KeyError:
            pass

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


async def is_path_exists(path: str) -> bool:
    """
    Check for the path exists or not asynchornously.
    """
    file_path = Path(path)
    return await run_in_threadpool(file_path.exists)


async def is_path_exists_and_has_file(path: str) -> bool:
    """
    Check for the path exists or not asynchornously.
    """
    folder_path = Path(path)
    path_exists = await run_in_threadpool(folder_path.exists)
    if not path_exists:
        return False

    # Check if the folder contains at least one file asynchronously
    files = list(await run_in_threadpool(folder_path.iterdir))

    return bool(files)


async def awrite_to_file(filename: str, content: str):
    async with aiofiles.open(filename, mode="w") as file:
        await file.write(content)


async def aread_file(file_path: str) -> str:
    """
    Async file read.
    """
    async with aiofiles.open(file_path, mode="r") as file:
        return await file.read()
