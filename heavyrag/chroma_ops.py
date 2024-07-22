import json
import os
from datetime import datetime

import chromadb

from heavyiq.config import get_config
from heavyrag.index import get_vectorstore
from heavyrag.logger import logger

CONFIG = get_config()


def backup_collection(collection: chromadb.Collection) -> str:
    # Get all data from the collection
    data = collection.get()

    # Create a backup file with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    backup_dir = CONFIG.rag_chromadb_collection_backup_dir
    backup_file = f"{backup_dir}/backup_{timestamp}.json"

    # create backup dir if not exists
    os.makedirs(backup_dir, exist_ok=True)

    # Write the data to the backup file
    with open(backup_file, "w") as file:
        json.dump(data, file)

    logger.info(f"Chromadb collection '{collection.name}' backup completed. Data saved to {backup_file}")
    return backup_file


def get_latest_backup_file(directory: str = CONFIG.rag_chromadb_collection_backup_dir) -> str:
    """
    Get the latest backup file.
    """
    backup_files = [f for f in os.listdir(directory) if f.startswith("backup_") and f.endswith(".json")]
    if not backup_files:
        raise FileNotFoundError("No backup files found.")
    latest_backup_file = max(backup_files, key=lambda x: os.path.getctime(os.path.join(directory, x)))
    return os.path.join(directory, latest_backup_file)


def restore_collection(new_collection: chromadb.Collection, backup_file: str, batch_size: int = 166):
    """
    Restore data from backup file to new chromadb collection.
    """
    # Read data from the backup file
    with open(backup_file, "r") as file:
        data = json.load(file)

    # Embed data with newer embed model and then put it to the chromadb collection
    # since the chroma_collection was created with the custom embedding function
    # it automatically re-embeds the documents if there wasn't any embeddings value passed to the below add method
    def batch():
        """
        Batch data into chunks of size n and then insert it to the chroma collection.
        Yield value should be like list[ids], list[metadatas], list[documents]
        """
        record_len = len(data["ids"])
        for ndx in range(0, record_len, batch_size):
            upper_limit = min(ndx + batch_size, record_len)
            yield data["ids"][ndx:upper_limit], data["metadatas"][ndx:upper_limit], data["documents"][ndx:upper_limit]

    for ids, metadatas, documents in batch():
        new_collection.add(ids=ids, metadatas=metadatas, documents=documents)

    logger.info(f"chromadb collection {new_collection.name} restore completed. Data loaded from {backup_file}")


def update_embeddings(collection: chromadb.Collection, batch_size: int = 166):
    """
    Helps to update only the embeddings.
    """
    backup_file = backup_collection(collection)
    collection_name = collection.name
    logger.info(f"Chromadb collection {collection_name} backedup successfully")
    # delete the collection
    collection._client.delete_collection(collection.name)
    # re-create the same collection
    new_collection = get_vectorstore(collection_name)._collection
    # restore data with new embeddings
    restore_collection(new_collection, backup_file, batch_size=batch_size)
