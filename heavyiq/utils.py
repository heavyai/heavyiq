import asyncio
import re
import threading
from collections.abc import Awaitable
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar
from urllib.parse import urlparse

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
    """
    Facade over the shared_state module for backward compatibility.
    
    This class delegates to heavyiq.shared_state which manages the actual
    shared state. When running under Gunicorn, shared_state uses a 
    multiprocessing Manager started by the master process. When running
    standalone, it uses a simple dict.
    
    Concurrency:
        - Individual get/put/delete operations are atomic (serialized by manager)
        - For compound operations (read-modify-write), use shared_state.lock()
    
    New code should use heavyiq.shared_state directly.
    """
    _instance: "SharedDictSingleton[KT, VT]" = None

    class Keys(Enum):
        HeavyDBLicenseEdition = "heavydb_license_edition"
        ConfFilePath = "config_file_path"

    def __new__(cls) -> "SharedDictSingleton":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def _shared_dict(self) -> dict:
        """Get the underlying shared dict from shared_state module."""
        from heavyiq import shared_state
        return shared_state.get_shared_dict()

    @classmethod
    def is_instantiated(cls: type["SharedDictSingleton"]) -> bool:
        return cls._instance is not None

    async def get(self, key: KT) -> Any:
        """Async get - individual operation is atomic."""
        return self._shared_dict.get(key)  # type: ignore

    async def put(self, key: KT, value: VT) -> None:
        """Async put - individual operation is atomic."""
        self._shared_dict[key] = value  # type: ignore

    async def delete(self, key: KT) -> None:
        """Async delete - individual operation is atomic."""
        try:
            del self._shared_dict[key]  # type: ignore
        except KeyError:
            pass

    def sget(self, key: KT) -> Any:
        """Sync get - individual operation is atomic."""
        return self._shared_dict.get(key)  # type: ignore

    def sput(self, key: KT, value: VT) -> None:
        """Sync put - individual operation is atomic."""
        self._shared_dict[key] = value  # type: ignore

    def sdelete(self, key: KT) -> None:
        """Sync delete - individual operation is atomic."""
        try:
            del self._shared_dict[key]  # type: ignore
        except KeyError:
            pass

    def get_last_schema_modification_check_time(self, table: str) -> float | None:
        """
        Retrieves the timestamp of the last schema modification check for a specified database table.
        """
        return self.sget(f"last_table_schema_change_check_{table}")  # type: ignore

    def put_last_schema_modification_check_time(self, table: str) -> None:
        """
        Updates the timestamp of the last schema modification check for a specified database table.
        """
        self.sput(f"last_table_schema_change_check_{table}", datetime.now().timestamp())  # type: ignore


class LRUCache(Generic[KT, VT]):
    """
    Implements shared LRU caching that can be accessed by any process.
    
    Uses shared_state module for cross-process sharing when running under Gunicorn,
    or simple Python containers when running standalone.
    """

    def __init__(
        self, 
        capacity: int = 100, 
        cache: dict | None = None,
        order: list | None = None,
    ) -> None:
        """
        Initialize LRU cache.
        
        Args:
            capacity: Maximum number of items in cache.
            cache: Optional pre-created dict (from shared_state.create_dict()).
                   If None, creates using shared_state.
            order: Optional pre-created list (from shared_state.create_list()).
                   If None, creates using shared_state.
        """
        from heavyiq import shared_state
        
        self.capacity: int = capacity
        self._lock: threading.Lock = threading.Lock()
        
        # Use provided containers or create new ones from shared_state
        self.cache = cache if cache is not None else shared_state.create_dict()
        self.order = order if order is not None else shared_state.create_list()

    def get(self, key: KT) -> Optional[VT]:
        """
        Returns the value of passed dict key if exists else return None.
        """
        with self._lock:
            if key in self.cache:
                # Move the key to the end (most recently used) in the order list
                self.move_to_end(key)
                return self.cache[key]
            return None

    def delete(self, key: KT) -> None:
        """
        Deletes the key and it's associated value from the cache dict.
        """
        with self._lock:
            try:
                del self.cache[key]
                self.order.remove(key)
            except KeyError:
                pass

    def get_key_starts_with(self, key_prefix: KT) -> list[KT]:
        """
        Gets the list of cache keys which startswith a particular prefix.
        This might be useful for getting all the keys relevant to a particular table
        which has the different caches stored based different params such as include_comments, include_top_k, etc.
        """
        keys_found = []
        for key in self.cache:
            if key.startswith(key_prefix):
                keys_found.append(key)

        return keys_found

    def get_key_contains(self, partial_key: str, key_prefix: KT | None = None) -> list[KT]:
        """
        Grab the keys which contain a particular string.
        """
        keys_found = []
        if key_prefix:
            matched_keys = self.get_key_starts_with(key_prefix)
        else:
            matched_keys = self.cache.keys()
        for key in matched_keys:
            if partial_key in key:  # type: ignore
                keys_found.append(key)

        return keys_found

    def delete_by_key_prefix(self, key_prefix: KT) -> None:
        """
        Deletes all the caches by key prefix.
        """
        for key in self.get_key_starts_with(key_prefix):
            self.delete(key)

    def get_recent_cache_by_key_prefix(self, key_prefix: KT) -> Optional[VT]:
        """
        Gets the most recent cache by key prefix.
        """
        keys = self.get_key_starts_with(key_prefix)
        if keys:
            return self.get(keys[-1])

        return None

    def put(self, key: KT, value: VT) -> None:
        """
        Helps to put the given key, value pair on the manager.Dict.
        """
        with self._lock:
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


class TablesCache(Generic[KT, VT]):
    """
    Helps to store schema related to multiple tables in LRUCache.
    
    Uses lazy initialization - the cache is created on first access,
    using shared_state for cross-process sharing.
    """

    _instance: "TablesCache[KT, VT] | None" = None
    _cache: "LRUCache[KT, VT] | None" = None
    _init_lock: threading.Lock = threading.Lock()
    _op_lock: threading.Lock = threading.Lock()
    key_prefix: str = "tables_cache"

    def __new__(cls: type["TablesCache"]) -> "TablesCache":
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def _get_cache(cls) -> "LRUCache[KT, VT]":
        """Lazy initialization of the cache."""
        if cls._cache is None:
            with cls._init_lock:
                if cls._cache is None:
                    cls._cache = LRUCache[KT, VT](capacity=200)
        return cls._cache

    @property
    def cache(self) -> "LRUCache[KT, VT]":
        return self._get_cache()

    @classmethod
    def initialize(cls) -> None:
        """
        Explicitly initialize the cache.
        Call this in app_initialize() before Gunicorn forks workers.
        """
        cls._get_cache()

    def create_cache_key_for_single_table(
        self,
        database: str,
        table: str,
        include_comments: bool = True,
        include_top_k: bool = True,
        include_timestamp: bool = True,
    ) -> str:
        """
        Form cache key for a specific table with different cache options.
        """
        return f"{self.key_prefix}:{database}:{table}:{include_comments}:{include_top_k}:{include_timestamp}"

    def decrypt_cache_key(self, cache_key: str) -> dict:
        """
        Decrypts the cache key.
        """
        _, database, table, include_comments, include_top_k, include_timestamp = cache_key.split(":")
        return {
            "database": database,
            "table": table,
            "include_comments": include_comments,
            "include_top_k": include_top_k,
            "include_timestamp": include_timestamp,
        }

    def delete(self, key: KT):
        """
        Helps to delete a particular cache.
        """
        with self._op_lock:
            self.cache.delete(key)

    def get(self, key: KT) -> VT | None:
        with self._op_lock:
            return self.cache.get(key)

    def put(self, key: KT, value: VT) -> None:
        with self._op_lock:
            return self.cache.put(key, value)

    def get_table_keys(self, database: str, table: str) -> list[str]:
        """
        Get all keys related to a table.
        """
        return self.cache.get_key_starts_with(f"{self.key_prefix}:{database}:{table}:")

    def delete_by_table_name(self, database: str, table_name: str):
        """
        Delete all the caches associated with a table name.
        """
        keys_found = self.get_table_keys(database=database, table=table_name)
        for key in keys_found:
            self.delete(key)  # type: ignore


def get_tables_cache() -> TablesCache[str, str]:
    """
    Get the global TablesCache instance (lazy initialization).
    
    This is the preferred way to access the tables cache.
    """
    # TablesCache is already a singleton via __new__
    return TablesCache[str, str]()


class _TablesCacheProxy:
    """
    Proxy class for backward compatibility with TABLES_CACHE module constant.
    
    Delegates all attribute access to the actual TablesCache singleton,
    which is lazily initialized on first access.
    """
    def __getattr__(self, name: str) -> Any:
        return getattr(get_tables_cache(), name)


# For backward compatibility - existing code uses TABLES_CACHE.get(), TABLES_CACHE.put(), etc.
TABLES_CACHE: TablesCache[str, str] = _TablesCacheProxy()  # type: ignore


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


async def semaphore_gather(num: int, coros: list[Awaitable], return_exceptions: bool = False) -> Any:
    semaphore = asyncio.Semaphore(num)

    async def _wrap_coro(coro: Awaitable) -> Any:
        async with semaphore:
            return await coro

    return await asyncio.gather(*(_wrap_coro(coro) for coro in coros), return_exceptions=return_exceptions)


def get_host_and_port(url: str) -> tuple[str, int]:
    parsed_url = urlparse(url)
    host = parsed_url.hostname
    port = parsed_url.port

    return host, port
