from __future__ import annotations
import re
import multiprocessing
from multiprocessing.managers import SyncManager
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Optional, Any, Iterable, TYPE_CHECKING, Callable, TypedDict
from copy import deepcopy

from heavyai import connect, Connection
from heavyiq.config import get_config
from heavyiq.utils import strip_sql_comments, is_destructive_sql, rate_sql_complexity, LRUCache

if TYPE_CHECKING:
    from heavydb._parsers import ColumnDetails


class PersistantConnection(Connection):
    """
    Helps to establish heavydb connection which persists for atleast an hour.

    Default heavyai.Connection gets closed once the corresponding __del__ method being called (ie. upon garbage collection).
    So we can't re-use the same session_id on `/query`, `/question` endpoints.

    This class helps to overcome the above issue, and the created session gets auto expire after certain limit
    (Default session-timeout-value set on server side is 60 mins).
    """

    def close(self):
        """Don't disconnect from the database. Let the session expire automatically."""
        self._closed = 1
        self._rbc = None


class StringLiteralOp(TypedDict):
    """
    A type representation for string literal operations.
    """

    operator: str
    literal: str
    database: str
    table: str
    column: str


class HeavyDB:
    """A heavydb database connection."""

    _manager: Optional[SyncManager] = None
    _top_k_cache: Optional[LRUCache[str, str]] = None
    _sample_rows_cache: Optional[LRUCache[str, str]] = None
    _table_schema_cache: Optional[LRUCache[str, str]] = None

    def __init__(
        self,
        conn: Connection,
        ignore_tables: Optional[list[str]] = None,
        include_tables: Optional[list[str]] = None,
        sample_rows_in_table_info: int = 2,
        custom_table_info: Optional[dict[str, str]] = None,
    ):
        if include_tables and ignore_tables:
            raise ValueError("Cannot specify both include_tables and ignore_tables")

        from heavyiq.logging_utils import get_heavyiq_logger

        self.logger = get_heavyiq_logger()
        self._conn = conn
        self.lock = Lock()

        self._all_tables = set(self._conn.get_tables())
        self._include_tables = set(include_tables) if include_tables else set()
        if self._include_tables:
            missing_tables = self._include_tables - self._all_tables
            if missing_tables:
                raise ValueError(f"include_tables {missing_tables} not found in database")
        self._ignore_tables = set(ignore_tables) if ignore_tables else set()
        if self._ignore_tables:
            missing_tables = self._ignore_tables - self._all_tables
            if missing_tables:
                print(f"WARNING: ignore_tables {missing_tables} not found in database")

        if not isinstance(sample_rows_in_table_info, int):
            raise TypeError("sample_rows_in_table_info must be an integer")

        self._sample_rows_in_table_info = sample_rows_in_table_info

        self._custom_table_info = custom_table_info
        if self._custom_table_info:
            if not isinstance(self._custom_table_info, dict):
                raise TypeError(
                    "table_info must be a dictionary with table names as keys and the desired table info as values"
                )
            # only keep the tables that are also present in the database
            intersection = set(self._custom_table_info).intersection(self._all_tables)
            self._custom_table_info = dict(
                (table, self._custom_table_info[table]) for table in self._custom_table_info if table in intersection
            )

    def __del__(self):
        try:
            self._conn.close()
        except Exception as e:
            print(f"Error: {e}")

    @classmethod
    def get_manager(cls) -> SyncManager:
        if cls._manager is None:
            cls._manager = multiprocessing.Manager()
        return cls._manager

    @classmethod
    def get_top_k_cache(cls) -> LRUCache[str, str]:
        if cls._top_k_cache is None:
            cls._top_k_cache = LRUCache[str, str](manager=cls.get_manager())
        return cls._top_k_cache

    @classmethod
    def get_sample_rows_cache(cls) -> LRUCache[str, str]:
        if cls._sample_rows_cache is None:
            cls._sample_rows_cache = LRUCache[str, str](manager=cls.get_manager())
        return cls._sample_rows_cache

    @classmethod
    def get_table_schema_cache(cls) -> LRUCache[str, str]:
        if cls._table_schema_cache is None:
            cls._table_schema_cache = LRUCache[str, str](manager=cls.get_manager())
        return cls._table_schema_cache

    @property
    def top_k_cache(self) -> LRUCache[str, str]:
        return self.get_top_k_cache()

    @property
    def sample_rows_cache(self) -> LRUCache[str, str]:
        return self.get_sample_rows_cache()

    @property
    def table_schema_cache(self) -> LRUCache[str, str]:
        return self.get_table_schema_cache()

    @classmethod
    def _connect_with_timeout(
        cls: type[HeavyDB], connect_func: Callable[[], Connection], timeout: float = 10
    ) -> Connection:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(connect_func)
            try:
                return future.result(timeout)
            except TimeoutError:
                raise Exception(f"HeavyDB failed to connect after {timeout} seconds")

    @classmethod
    def from_uri(cls: type[HeavyDB], database_uri: str, **kwargs: Any) -> HeavyDB:
        """Create a database connection from a database URI."""
        conn = cls._connect_with_timeout(lambda: connect(database_uri), kwargs.pop("timeout", 10))
        return cls(conn, **kwargs)

    @classmethod
    def from_env(cls: type[HeavyDB], db_name: Optional[str] = None, **kwargs: Any) -> HeavyDB:
        """Create a database connection from environment variables."""
        config = get_config()
        if not config.heavydb_username or not config.heavydb_password or not (db_name or config.heavydb_dbname):
            raise ValueError("Please set the config variables heavydb_username, heavydb_password and heavydb_dbname")

        def connect_func() -> Connection:
            return connect(
                user=config.heavydb_username,
                password=config.heavydb_password,
                host=config.heavydb_host,
                port=config.heavydb_port,
                dbname=db_name or config.heavydb_dbname,
                protocol=config.heavydb_protocol,
            )

        conn = cls._connect_with_timeout(connect_func, kwargs.pop("timeout", 10))
        return cls(conn, **kwargs)

    @classmethod
    def from_session(cls: type[HeavyDB], session_id: str, **kwargs: Any) -> HeavyDB:
        """Create a database connection from a session id."""
        config = get_config()

        def connect_func() -> Connection:
            return connect(
                sessionid=session_id,
                host=config.heavydb_host,
                port=config.heavydb_port,
                protocol=config.heavydb_protocol,
            )

        conn = cls._connect_with_timeout(connect_func, kwargs.pop("timeout", 10))
        return cls(conn, **kwargs)

    @classmethod
    def from_creds(
        cls: type[HeavyDB], username: str, password: str, dbname: Optional[str] = None, **kwargs: Any
    ) -> HeavyDB:
        """Create a database connection from username and password."""
        config = get_config()
        if not config.heavydb_dbname and not dbname:
            raise ValueError(
                "Please set the config variable heavydb_dbname or provide a dbname as an argument to HeavyDB.from_creds"
            )
        conn: Connection = connect(
            user=username,
            password=password,
            host=config.heavydb_host,
            port=config.heavydb_port,
            protocol=config.heavydb_protocol,
            dbname=dbname or config.heavydb_dbname,
        )
        return cls(conn, **kwargs)

    @property
    def dialect(self) -> str:
        """Return string representation of dialect to use."""
        return "ANSI SQL"

    def get_usable_table_names(self) -> Iterable[str]:
        """Get names of tables available."""
        if self._include_tables:
            return self._include_tables
        return self._all_tables - self._ignore_tables

    @classmethod
    def create_session_id(cls: type["HeavyDB"], db_name: str | None = None) -> str:
        """
        Create a new db session and return it's id.
        """
        config = get_config()
        if not config.heavydb_username or not config.heavydb_password:
            raise ValueError("Please set the config variables heavydb_username and heavydb_password")

        conn: PersistantConnection = PersistantConnection(
            user=config.heavydb_username,
            password=config.heavydb_password,
            host=config.heavydb_host,
            port=config.heavydb_port,
            protocol=config.heavydb_protocol,
            dbname=db_name or config.heavydb_dbname,
        )
        return conn._session

    @property
    def table_info(self) -> str:
        """Information about all usable tables in the database."""
        return self.get_table_info()

    def get_table_columns(self, table: str) -> list[ColumnDetails]:
        """Get details about the columns in a table."""
        self.logger.debug(f"Getting columns for table {table}")
        with self.lock:
            table_details = self._conn.get_table_details(table)
            self.logger.debug(f"Got columns for table {table}")
            return table_details

    def get_table_schema(self, table: str) -> str:
        self.logger.debug(f"Getting schema for table {table}")
        cache_key = f"{self._conn._dbname}.{table}"
        cached_value = self.table_schema_cache.get(cache_key)
        if cached_value is not None:
            self.logger.debug(f"Got schema for table {table} from cache")
            return cached_value
        """Get the schema of a table."""
        create_command = f"SHOW CREATE TABLE {table};"
        with self.lock:
            cursor = self._conn.execute(create_command)
        table_schema = cursor.fetchone()[0]  # type: ignore
        table_schema = re.sub(r" ENCODING .*\)([,\)])", r"\1", table_schema)  # type: ignore
        table_schema = re.sub(r",\n.*SHARED DICTIONARY.*REFERENCES.*\([A-Za-z0-9_]*\)", "", table_schema)
        table_schema = re.sub(r"\n", "", table_schema)
        if "WITH (" in table_schema:
            table_schema = table_schema[: table_schema.index("WITH (")]
        self.table_schema_cache.put(cache_key, table_schema)
        self.logger.debug(f"Got schema for table {table}")
        return table_schema

    def validate_query(self, query: str) -> list:
        """Validate a query."""
        # Remove block comments
        query = strip_sql_comments(query)
        if is_destructive_sql(query):
            raise ValueError("Destructive SQL is not allowed")
        if "::" in query:
            raise ValueError("Double colon cast syntax is not allowed. Use CAST() instead.")
        with self.lock:
            return self._conn._client.sql_validate(self._conn._session, query)

    def get_column_top_k(self, table: str, column: str, _k: int = 5) -> tuple[Optional[list[str]], bool]:
        config = get_config()
        cardinality_threshold = config.column_top_k_cardinality_threshold
        high_cardinality_sample = config.column_top_k_high_cardinality_sample

        # We need to ensure that column names that are reserved keywords are double quoted otherwise an error will occur
        # Todo (todd): These are only a partial list of reserved keywords, ensure we have an exhaustive list
        reserved_keywords = [
            "LANGUAGE",
            "RANK",
            "RESULT",
            "DATE",
            "TIMESTAMP",
            "LENGTH",
            "YEAR",
            "QUARTER",
            "MONTH",
            "WEEK",
            "DAY",
        ]
        if column.upper() in reserved_keywords:
            column = f'"{column}"'
        # check to see if the column is low cardinality
        # fetch top (threshold + 1)
        # if there are < (threshold + 1) values, it's low cardinality and we can return all of them
        # if there are >= (threshold + 1) values, it's high cardinality and we need to sample the top high_cardinality_sample
        """Get the top k values for a column."""
        self.logger.debug(f"Getting top k values for column {column} in table {table}")
        top_k_statement = f"SELECT {column}, COUNT(*) as cnt FROM {table} WHERE {column} is not null GROUP BY {column} ORDER BY cnt DESC LIMIT {cardinality_threshold + 1};"
        with self.lock:
            cursor = self._conn.execute(top_k_statement)
        top_k_res: list[str] = [str(v[0]) for v in cursor.fetchall()]
        is_high_cardinality = len(top_k_res) > cardinality_threshold
        if is_high_cardinality:
            # high-cardinality, return sample
            self.logger.debug(f"Column {column} is high-cardinality. Returning top {high_cardinality_sample} values")
            top_k_res = top_k_res[:high_cardinality_sample]
        else:
            self.logger.debug(f"Column {column} is low-cardinality.")
        self.logger.debug(f"Got top k values for column {column} in table {table}")
        if not any([v for v in top_k_res if v.startswith("MULTIPOLYGON")]):
            return top_k_res, is_high_cardinality
        return None, is_high_cardinality

    def get_sample_rows(self, table_name: str) -> str:
        self.logger.debug(f"Getting sample rows for table {table_name}")
        cache_key = f"{self._conn._dbname}.{table_name}"
        cached_value = self.sample_rows_cache.get(cache_key)
        if cached_value is not None:
            self.logger.debug(f"Got sample rows for table {table_name} from cache")
            return cached_value
        # build the select command
        command = f"SELECT * FROM {table_name} LIMIT {self._sample_rows_in_table_info}"

        # save the columns in string format
        columns_str = ",".join([col.name for col in self.get_table_columns(table_name)])

        # get the sample rows
        with self.lock:
            sample_rows = self._conn.execute(command)
        # shorten values in the sample rows
        sample_rows = list(map(lambda ls: [str(i)[:100] for i in ls], sample_rows))

        # save the sample rows in string format
        sample_rows_str = "\n".join([",".join(row) for row in sample_rows])

        res = f"{self._sample_rows_in_table_info} rows from {table_name} table:\n{columns_str}\n{sample_rows_str}"

        self.sample_rows_cache.put(cache_key, res)
        self.logger.debug(f"Got sample rows for table {table_name}")
        return res

    def get_top_k(self, table_name: str) -> str:
        self.logger.debug(f"Getting top k values for table {table_name}")
        cache_key = f"{self._conn._dbname}.{table_name}"
        cached_value = self.top_k_cache.get(cache_key)
        if cached_value is not None:
            self.logger.debug(f"Got top k values for table {table_name} from cache")
            return cached_value
        text_columns = [
            c.name
            for c in self.get_table_columns(table_name)
            if c.type == "STR" and c.encoding == "DICT" and c.is_array is False
        ]
        low_cardinality_columns = []
        high_cardinality_columns = []
        for col in text_columns:
            top_k_res, is_high_cardinality = self.get_column_top_k(table_name, col)
            if top_k_res:
                if is_high_cardinality:
                    high_cardinality_columns.append((col, top_k_res))
                else:
                    low_cardinality_columns.append((col, top_k_res))
        top_k_strings = ""
        if low_cardinality_columns:
            top_k_strings += "Low cardinality columns and every possible value:\n"
            for col, top_k_res in low_cardinality_columns:
                top_k_strings += f"{col}: {', '.join(top_k_res)}\n"
        if high_cardinality_columns:
            top_k_strings += "High cardinality columns and most common values:\n"
            for col, top_k_res in high_cardinality_columns:
                top_k_strings += f"{col}: {', '.join(top_k_res)}\n"
        self.top_k_cache.put(cache_key, top_k_strings)
        self.logger.debug(f"Got top k values for table {table_name}")
        return top_k_strings

    def get_single_table_info(self, table_name: str, include_samples: bool = True, include_top_k: bool = True) -> str:
        """Get information about a single table."""
        all_table_names = self.get_usable_table_names()
        if table_name not in all_table_names:
            raise ValueError(f"table_name {table_name} not found in database")

        if self._custom_table_info and table_name in self._custom_table_info:
            return self._custom_table_info[table_name]

        parts = []

        parts.append(self.get_table_schema(table_name))

        if include_samples and self._sample_rows_in_table_info:
            parts.append(self.get_sample_rows(table_name))

        if include_top_k:
            parts.append(self.get_top_k(table_name))

        return "\n".join(parts)

    def get_table_info(self, table_names: Optional[list[str]] = None, **kwargs) -> str:
        """Get information about specified tables.
        Follows best practices as specified in: Rajkumar et al, 2022
        (https://arxiv.org/abs/2204.00498)
        If `sample_rows_in_table_info`, the specified number of sample rows will be
        appended to each table description. This can increase performance as
        demonstrated in the paper.
        """
        all_table_names = self.get_usable_table_names()
        if table_names is not None:
            missing_tables = set(table_names).difference(all_table_names)
            if missing_tables:
                raise ValueError(f"table_names {missing_tables} not found in database")
            all_table_names = table_names

        tables = []
        for table in all_table_names:
            tables.append(self.get_single_table_info(table, **kwargs))

        final_str = "\n\n".join(tables)
        return final_str

    def run(self, command: str, fetch: str = "all", to_str: bool = True) -> str | tuple | list:
        """Execute a SQL command and return a string representing the results.
        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.
        If to_str is True (by default), result should be converted to string.
        If fetch = one and to_str = False, then a tuple will be returned.
        If fetch = all and to_str = False, then a list of tuples will be returned.
        """
        command = strip_sql_comments(command)
        if is_destructive_sql(command):
            raise ValueError("Destructive SQL is not allowed")
        with self.lock:
            cursor = self._conn.execute(command)
        if fetch == "all":
            result = cursor.fetchall()
        elif fetch == "one":
            result = cursor.fetchone()
        else:
            raise ValueError("Fetch parameter must be either 'one' or 'all'")
        if to_str:
            return str(result)
        return result  # type: ignore

    def get_query_plan(self, query: str) -> str:
        query = strip_sql_comments(query)
        if is_destructive_sql(query):
            raise ValueError("Destructive SQL is not allowed")
        with self.lock:
            cursor = self._conn.execute(f"EXPLAIN plan {query}")
        result: tuple[str] = cursor.fetchone()  # type: ignore
        return str(result[0])

    def get_detailed_query_plan(self, query: str) -> str:
        query = strip_sql_comments(query)
        if is_destructive_sql(query):
            raise ValueError("Destructive SQL is not allowed")
        with self.lock:
            cursor = self._conn.execute(f"EXPLAIN CALCITE DETAILED {query}")
        query_plan: tuple[str] = cursor.fetchone()  # type: ignore
        return str(query_plan[0])

    def extract_column_mappings(self, detailed_query_plan: str) -> dict[str, tuple[str, str, str]]:
        self.logger.debug(f"Extracting column mappings from query plan: {detailed_query_plan}")
        result = {}
        # Use a regex pattern to capture (database, table, column) and literal values in LogicalFilter
        pattern = r"\[\$([0-9]+)->db:([\w]+),tableName:([\w]+),colName:([\w]+)\]"
        matches = re.findall(pattern, detailed_query_plan)
        for id, db, table, column in matches:
            col_id = (db, table, column)
            result[id] = col_id

        self.logger.debug("Finished extracting column mappings")
        return result

    def extract_string_literal_ops(self, detailed_query_plan: str) -> dict[str, tuple[str, str]]:
        self.logger.debug(f"Extracting string literal operations from query plan: {detailed_query_plan}")
        result = {}
        pattern1 = r"(LIKE|PG_ILIKE|>=|<=|<>|=)\(\$(\d+), '([\w\- ]+)'"
        pattern2 = r"(LIKE|PG_ILIKE|>=|<=|<>|=)\('([\w\- ]+)', \$(\d+)\)"

        matches1 = re.findall(pattern1, detailed_query_plan)
        matches2 = re.findall(pattern2, detailed_query_plan)
        for op, id, literal in matches1:
            result[id] = (op, literal)
        for op, literal, id in matches2:
            result[id] = (op, literal)

        self.logger.debug("Finished extracting string literal operations")
        return result

    def get_string_literal_ops(self, query: str) -> list[StringLiteralOp]:
        detailed_query_plan = self.get_detailed_query_plan(query)
        col_mapping = self.extract_column_mappings(detailed_query_plan)
        str_literal_ops = self.extract_string_literal_ops(detailed_query_plan)
        result: list[StringLiteralOp] = []
        for id, (op, literal) in str_literal_ops.items():
            db, table, column = col_mapping[id]
            result.append({"operator": op, "literal": literal, "database": db, "table": table, "column": column})
        return result

    def correct_string_literal(self, literal: StringLiteralOp, exact_match_threshold: float) -> StringLiteralOp:
        """
        Tries to correct a given string literal by searching for close matches in a database.

        Parameters
        ----------
        literal : StringLiteralOp
            A dictionary representing the string literal to be corrected.
            Expected keys are 'column', 'database', 'table', 'literal', and 'operator'.

        exact_match_threshold : float
            A threshold value between 0 and 1 that determines how strict the exact match ratio should be
            before deciding to change the literal or operator. If the exact match ratio is below this
            threshold, the operator might be changed to "ILIKE" or "NOT ILIKE".
        """
        self.logger.debug(f"Correcting string literal: {literal['column']} : {literal['literal']}")
        case_match_query = f"SELECT {literal['column']}, COUNT(*) FROM {literal['database']}.{literal['table']} WHERE {literal['column']} ILIKE '{literal['literal']}' GROUP BY {literal['column']} ORDER BY COUNT(*) DESC;"
        with self.lock:
            cursor = self._conn.execute(case_match_query)
        case_match_rows = cursor.fetchall()

        num_case_match_rows = len(case_match_rows)
        exact_match_count = 0
        total_count = 0
        altered_literal = deepcopy(literal)

        for row in case_match_rows:
            if row[0] == literal["literal"]:
                exact_match_count += row[1]  # type: ignore
            total_count += row[1]  # type: ignore

        if total_count > 0 and literal["operator"] != "ILIKE":
            if exact_match_count == 0 and num_case_match_rows == 1:
                altered_literal["literal"] = str(case_match_rows[0][0])
                return literal
            elif exact_match_count / total_count < exact_match_threshold:
                if literal["operator"] == "<>":
                    altered_literal["operator"] = "NOT ILIKE"
                else:
                    altered_literal["operator"] = "ILIKE"
                return altered_literal
        elif total_count == 0:
            lower_literal = literal["literal"].lower()
            similarity_query = f"WITH distinct_values AS (SELECT LOWER({literal['column']}) AS lower_attr, COUNT(*) AS num_str_values FROM {literal['database']}.{literal['table']} GROUP BY LOWER({literal['column']})) SELECT lower_attr, LEVENSHTEIN_DISTANCE(lower_attr, '{lower_literal}') - ABS(LENGTH(lower_attr) - LENGTH('{lower_literal}')) FROM distinct_values WHERE LEVENSHTEIN_DISTANCE(lower_attr, '{lower_literal}') - ABS(LENGTH(lower_attr) - LENGTH('{lower_literal}')) < 5 ORDER BY LEVENSHTEIN_DISTANCE(lower_attr, '{lower_literal}') - ABS(LENGTH(lower_attr) - LENGTH('{lower_literal}')) ASC, num_str_values DESC LIMIT 2;"

            with self.lock:
                cursor = self._conn.execute(similarity_query)
            similarity_rows = cursor.fetchall()

            num_similarity_rows = len(similarity_rows)

            if num_similarity_rows > 0:
                if (
                    num_similarity_rows > 1
                    and similarity_rows[0][1] == 0
                    and similarity_rows[0][1] == similarity_rows[1][1]
                ):
                    # Here there are at least two matches such that the user-provided literal is a full substring of
                    # the column value. In this case, we will match against all strings that our string literal
                    # is a substring of
                    altered_literal["literal"] = f"%{lower_literal}%"
                else:
                    # There was only one match, or two matches and at least one did have a 0 distance score, so pick the
                    # top returned value (we've sorted in ascending order by score and descending order by number
                    # of string matches)
                    altered_literal["literal"] = str(similarity_rows[0][0])
                if literal["operator"] == "<>":
                    altered_literal["operator"] = "NOT ILIKE"
                else:
                    altered_literal["operator"] = "ILIKE"
            return altered_literal

        return altered_literal

    def correct_string_literals(self, query: str, exact_match_threshold: float = 0.999999) -> str:
        config = get_config()
        if not config.enable_str_literal_correction:
            return query

        self.logger.debug(f"Correcting string literals in query: {query}")
        altered_query = deepcopy(query)
        try:
            str_literal_ops_list = self.get_string_literal_ops(query)
            if len(str_literal_ops_list) == 0:
                self.logger.debug("No string literals found in query")
                return query
            for str_literal_op in str_literal_ops_list:
                altered_str_literal_op = self.correct_string_literal(str_literal_op, exact_match_threshold)
                if altered_str_literal_op != str_literal_op:
                    if altered_str_literal_op["operator"] != str_literal_op["operator"]:
                        altered_query = altered_query.replace(
                            f"{str_literal_op['column']} {str_literal_op['operator']} '{str_literal_op['literal']}'",
                            f"{str_literal_op['column']} {altered_str_literal_op['operator']} '{altered_str_literal_op['literal']}'",
                        )
                        self.logger.debug(
                            f"Operator changed from {str_literal_op['operator']} to {altered_str_literal_op['operator']}"
                        )
                    if altered_str_literal_op["literal"] != str_literal_op["literal"]:
                        altered_query = altered_query.replace(
                            f"{str_literal_op['column']} {altered_str_literal_op['operator']} '{str_literal_op['literal']}'",
                            f"{str_literal_op['column']} {altered_str_literal_op['operator']} '{altered_str_literal_op['literal']}'",
                        )
                        self.logger.debug(
                            f"Literal changed from {str_literal_op['literal']} to {altered_str_literal_op['literal']}"
                        )
        except Exception as e:
            self.logger.info(f"Error correcting string literals: {e}")
            return query

        self.logger.debug(f"Altered query: {altered_query}")
        return altered_query

    def complexity(self, command: str) -> int:
        command = strip_sql_comments(command)
        plan = self.get_query_plan(command)
        return rate_sql_complexity(plan)

    def get_table_info_no_throw(self, table_names: Optional[list[str]] = None) -> str:
        """Get information about specified tables.
        Follows best practices as specified in: Rajkumar et al, 2022
        (https://arxiv.org/abs/2204.00498)
        If `sample_rows_in_table_info`, the specified number of sample rows will be
        appended to each table description. This can increase performance as
        demonstrated in the paper.
        """
        try:
            return self.get_table_info(table_names)
        except ValueError as e:
            """Format the error message"""
            return f"Error: {e}"

    def run_no_throw(self, command: str, fetch: str = "all") -> str:
        """Execute a SQL command and return a string representing the results.
        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.
        If the statement throws an error, the error message is returned.
        """
        try:
            return self.run(command, fetch)
        except Exception as e:
            """Format the error message"""
            return f"Error: {e}"
