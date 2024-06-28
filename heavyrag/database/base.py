from heavyiq.config import get_config

from .database import Database

config = get_config()

ragdb = Database(config.rag_database_uri)
