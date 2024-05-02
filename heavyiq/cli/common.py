import asyncio

from heavyiq.langchain.heavydb import HeavyDB


class HeavyDBConnectionPool:
    _connections: dict[str, HeavyDB] = {}
    _lock: asyncio.Lock = asyncio.Lock()

    @classmethod
    async def get_connection(cls: type["HeavyDBConnectionPool"], database: str) -> HeavyDB | None:
        if database in cls._connections:
            return cls._connections[database]
        async with cls._lock:
            if database in cls._connections:
                return cls._connections[database]
            conn = await cls.create_connection(database)
            if conn:
                cls._connections[database] = conn
            return conn

    @classmethod
    async def create_connection(cls: type["HeavyDBConnectionPool"], database: str) -> HeavyDB | None:
        try:
            print(f"Creating new connection for {database}")
            return await HeavyDB.create_with_persistant_connection_async(db_name=database)
        except Exception:
            print(f"Failed to create persistant heavydb connection for {database} database.")
            return None

    @classmethod
    async def refresh_connection(cls: type["HeavyDBConnectionPool"], database: str) -> HeavyDB:
        print(f"Refreshing connection for database {database}")
        cls._connections.pop(database, None)
        return await cls.get_connection(database)
