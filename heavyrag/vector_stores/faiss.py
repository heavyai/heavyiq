import fcntl
import os
import pickle
import sys
import threading
import time
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any, Union, cast

import faiss
import numpy as np
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.types import (
    ExactMatchFilter,
    FilterCondition,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from llama_index.vector_stores.faiss import FaissVectorStore

from heavyrag.logger import logger
from heavyrag.utils import uuid4_hex_to_int64

# On macOS, limit FAISS threads to prevent gunicorn worker crashes
if sys.platform == "darwin":
    faiss.omp_set_num_threads(1)

# Track last update time for reload optimization
last_update_time = 0

# Global cached index (per-process)
faiss_index = None


class FaissFileLock:
    """
    File-based lock for FAISS index operations.
    Works across separate processes (e.g., gunicorn workers).
    Supports both shared (read) and exclusive (write) locks.
    """

    def __init__(self, lock_file: str, timeout: float = 30.0):
        """
        Initialize the file lock.
        
        Args:
            lock_file: Path to the lock file (will be created if doesn't exist)
            timeout: Maximum time to wait for lock acquisition (seconds)
        """
        self.lock_file = lock_file
        self.timeout = timeout
        self._fd = None
        
    def _ensure_lock_file(self):
        """Ensure the lock file exists."""
        lock_dir = os.path.dirname(self.lock_file)
        if lock_dir and not os.path.exists(lock_dir):
            os.makedirs(lock_dir, exist_ok=True)
        # Create lock file if it doesn't exist
        if not os.path.exists(self.lock_file):
            open(self.lock_file, 'a').close()

    @contextmanager
    def read_lock(self):
        """
        Acquire a shared (read) lock. Multiple readers can hold the lock simultaneously.
        """
        self._ensure_lock_file()
        fd = open(self.lock_file, 'r')
        try:
            start_time = time.time()
            while True:
                try:
                    fcntl.flock(fd.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.time() - start_time > self.timeout:
                        raise TimeoutError(f"Could not acquire read lock on {self.lock_file} within {self.timeout}s")
                    time.sleep(0.01)
            yield
        finally:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
            fd.close()

    @contextmanager
    def write_lock(self):
        """
        Acquire an exclusive (write) lock. Only one writer can hold the lock.
        Blocks all readers and other writers.
        """
        self._ensure_lock_file()
        fd = open(self.lock_file, 'r+')
        try:
            start_time = time.time()
            while True:
                try:
                    fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.time() - start_time > self.timeout:
                        raise TimeoutError(f"Could not acquire write lock on {self.lock_file} within {self.timeout}s")
                    time.sleep(0.01)
            yield
        finally:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
            fd.close()


# Global file lock instance (will be initialized per persist_dir)
_file_locks: dict[str, FaissFileLock] = {}
_file_locks_lock = threading.Lock()


def get_file_lock(persist_dir: str) -> FaissFileLock:
    """Get or create a file lock for the given persist directory."""
    with _file_locks_lock:
        if persist_dir not in _file_locks:
            lock_file = os.path.join(persist_dir, ".faiss.lock")
            _file_locks[persist_dir] = FaissFileLock(lock_file)
        return _file_locks[persist_dir]


def get_faiss_index(persist_dir: str | None = None, dimension: int = 1024):
    """
    Get or create the FAISS index. Uses file lock for thread-safe initialization.
    """
    global faiss_index
    if faiss_index:
        return faiss_index
    
    # Use file lock for initialization to prevent race conditions
    file_lock = get_file_lock(persist_dir)
    with file_lock.read_lock():
        # Double-check after acquiring lock
        if faiss_index:
            return faiss_index
        logger.info("Reading faiss index from disk")
        _index_file = os.path.join(persist_dir, "faiss_index.bin")
        if os.path.exists(_index_file):
            faiss_index = faiss.read_index(_index_file, faiss.IO_FLAG_MMAP)
        else:
            faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
        return faiss_index


class FaissIQVectorStore(FaissVectorStore):
    """
    Faiss vector store wrapper class with file-based locking for concurrent access.
    """

    _persis_dir: str = PrivateAttr()
    _metadata_store: dict = PrivateAttr()
    _index_file: str = PrivateAttr()
    _metadata_file: str = PrivateAttr()
    _file_lock: FaissFileLock = PrivateAttr()

    def __init__(
        self,
        persist_dir: str,
        dimension: int = 1024,
        index_file: str = "faiss_index.bin",
        metadata_file: str = "metadata.pkl",
    ):
        """
        Initialize the FAISS index. Load from file if it exists, or create a new index otherwise.

        Args:
            dimension: The dimension of the embeddings.
            index_file: The file path for storing/loading the FAISS index.
            metadata_file: The file path for storing/loading the metadata.
        """
        os.makedirs(persist_dir, exist_ok=True)  # create the persist folder

        index = get_faiss_index(persist_dir, dimension=dimension)

        # Call super().__init__ first, then set private attributes
        # (Pydantic v2 resets __pydantic_private__ during __init__)
        super().__init__(faiss_index=index)

        # Now set private attributes after super().__init__
        self._persis_dir = persist_dir
        self._metadata_store = {}
        self._index_file = os.path.join(persist_dir, index_file)
        self._metadata_file = os.path.join(persist_dir, metadata_file)
        self._file_lock = get_file_lock(persist_dir)

        self.load_metadata()

    def load_metadata(self, use_file_lock: bool = False):
        """
        Load the metadata from the metadata file.
        
        Args:
            use_file_lock: Whether to acquire file lock (False when called from within locked context)
        """
        def _load():
            if os.path.exists(self._metadata_file):
                logger.info(f"Loading metadata from {self._metadata_file}")
                with open(self._metadata_file, "rb") as f:
                    self._metadata_store = pickle.load(f)
            else:
                logger.info("No metadata file found, initializing empty metadata store.")
                self._metadata_store = {}
        
        if use_file_lock:
            with self._file_lock.read_lock():
                _load()
        else:
            _load()

    def _reload_internal(self):
        """
        Internal reload - caller must hold appropriate file lock.
        """
        global last_update_time
        logger.debug("Started reloading faiss index and metadata.")
        
        try:
            if os.path.exists(self._index_file):
                file_mod_time = os.path.getmtime(self._index_file)
                # Reload the index only if it has been modified since the last reload
                if file_mod_time > last_update_time:
                    self._faiss_index = faiss.read_index(self._index_file)
                    self.load_metadata(use_file_lock=False)
                    last_update_time = file_mod_time
                    logger.debug(f"FAISS index reloaded by process {os.getpid()} at {file_mod_time}")
                else:
                    logger.debug("No need to reload FAISS index; no changes detected.")
            else:
                logger.debug(f"FAISS index file not found at {self._index_file}")
        except Exception as e:
            logger.error(f"Error reloading FAISS index: {e}")

    def reload(self):
        """
        Reload faiss index and metadata from disk with proper file locking.
        Uses file-based locking to coordinate across separate processes.
        """
        with self._file_lock.read_lock():
            self._reload_internal()

    def remove(self, ids: list[int] | None = None, node_ids: list[str] | None = None) -> None:
        """
        Remove vectors from the FAISS index as well as from metadata dump using their IDs.
        Supports both node_ids(str) as well as the vector store index ids (int).
        Uses write lock for the entire operation to ensure atomicity.
        """
        with self._file_lock.write_lock():
            self._reload_internal()
            
            if ids:
                ids_to_remove = np.array(ids, dtype=np.int64)
            elif node_ids:
                ids_to_remove = np.array([uuid4_hex_to_int64(id_) for id_ in node_ids], dtype=np.int64)
            else:
                raise ValueError("ids or node_ids need to be passed!")
            
            # Remove the vectors based on their IDs
            self._faiss_index.remove_ids(ids_to_remove)

            # Remove the corresponding metadata from the metadata store
            for id_ in ids_to_remove:
                if id_ in self._metadata_store:
                    del self._metadata_store[id_]

            self._persist_internal()

    def reset(self):
        """
        Deletes all the index data as well as the metadata.
        Uses write lock for the entire operation.
        """
        with self._file_lock.write_lock():
            self._faiss_index.reset()
            self._metadata_store = {}
            self._persist_internal()

    def get_ids(self) -> list:
        """
        List ids exists on the index.
        """
        return faiss.vector_to_array(self._faiss_index.id_map).tolist()

    def add(
        self,
        nodes: Sequence[BaseNode],
        **add_kwargs: Any,
    ) -> list[int]:
        """Add nodes to index.

        NOTE: in the Faiss vector store, we do not store text in Faiss.
        Uses write lock for the entire operation to ensure atomicity across processes.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        """
        with self._file_lock.write_lock():
            self._reload_internal()
            
            new_ids = []
            for node in nodes:
                text_embedding = node.get_embedding()
                text_embedding_np = np.array(text_embedding, dtype="float32").reshape(1, -1)
                new_id = uuid4_hex_to_int64(node.node_id)
                self._faiss_index.add_with_ids(text_embedding_np, np.array([new_id], dtype=np.int64))
                new_ids.append(new_id)
                metadata = {**node.metadata, "id": node.node_id}
                self._metadata_store[new_id] = {"text": node.get_content(), "metadata": metadata}
            
            self._persist_internal()
            return new_ids

    def apply_filter(self, metadata: dict, filter: Union[MetadataFilter, ExactMatchFilter, MetadataFilters]) -> bool:
        """
        Apply a single filter (or nested filter) to the metadata.

        Args:
            metadata: The document's metadata.
            filter: A MetadataFilter, ExactMatchFilter, or nested MetadataFilters.

        Returns:
            bool: True if the metadata satisfies the filter.
        """
        if isinstance(filter, ExactMatchFilter):
            # Apply exact match filter
            return metadata.get(filter.key) == filter.value
        elif isinstance(filter, MetadataFilter):
            # Apply other operators like '>', '<', '>=', etc.
            metadata_value = metadata.get(filter.key)
            if filter.operator == ">":
                return metadata_value > filter.value
            elif filter.operator == "<":
                return metadata_value < filter.value
            elif filter.operator == ">=":
                return metadata_value >= filter.value
            elif filter.operator == "<=":
                return metadata_value <= filter.value
            elif filter.operator == "!=":
                return metadata_value != filter.value
            # Add more operators as needed
            else:
                return False
        elif isinstance(filter, MetadataFilters):
            # Nested MetadataFilters: combine the results based on the condition (AND/OR)
            if filter.condition == FilterCondition.AND:
                return all(self.apply_filter(metadata, sub_filter) for sub_filter in filter.filters)
            elif filter.condition == FilterCondition.OR:
                return any(self.apply_filter(metadata, sub_filter) for sub_filter in filter.filters)
        return False

    def get_embedding_by_id(self, doc_id: str) -> np.ndarray:
        """Retrieve the embedding for a given document ID (string converted to int)."""
        # Convert the doc_id from string to int if it's a string
        int_doc_id = int(doc_id)

        if int_doc_id >= self._faiss_index.ntotal:
            raise ValueError(f"doc_id {int_doc_id} out of bounds. Index contains {self._faiss_index.ntotal} elements.")

        # Use FAISS's reconstruct method to retrieve the embedding
        embedding = self._faiss_index.reconstruct(int_doc_id)
        return embedding

    def create_node(self, metadata: dict) -> BaseNode:
        """
        Create a TextNode from metadata dict. metadata dict must contain text, metadata keys.
        """
        node_metadata = metadata.get("metadata", {})
        node = TextNode(text=metadata.get("text", ""), metadata=node_metadata, id_=node_metadata["id"])  # type: ignore
        return node

    def load_and_search(self, query_vector: np.array, top_k: int, filtered_ids: list[int] | None = None):
        """
        Perform FAISS search with file-based read lock for cross-process safety.
        """
        with self._file_lock.read_lock():
            if filtered_ids:
                id_selector = faiss.IDSelectorBatch(np.array(filtered_ids, dtype=np.int64))
                search_params = faiss.SearchParameters()
                search_params.selector = id_selector
                distances, indices = self._faiss_index.search(query_vector, top_k, params=search_params)
            else:
                distances, indices = self._faiss_index.search(query_vector, top_k)

            return distances, indices

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Perform a FAISS similarity search with optional metadata filtering using complex MetadataFilters.

        Args:
            query: VectorStoreQuery object that includes query_embedding, and_filters, or_filters, and top_k.
            kwargs: Additional keyword arguments.

        Returns:
            VectorStoreQueryResult: The result object containing nodes, similarities, and ids.
        """
        self.reload()
        query_embedding = query.query_embedding
        top_k = query.similarity_top_k or 5
        metadata_filters = query.filters

        filtered_ids = []
        id_to_metadata = {}

        # Step 1: Filter documents based on MetadataFilters
        for doc_id, metadata in self._metadata_store.items():
            if metadata_filters and isinstance(metadata_filters, MetadataFilters):
                # Apply the filters recursively
                if self.apply_filter(metadata.get("metadata", {}), metadata_filters):
                    filtered_ids.append(doc_id)
                    id_to_metadata[doc_id] = metadata
            else:
                # No filters applied, include all documents
                filtered_ids.append(doc_id)
                id_to_metadata[doc_id] = metadata

        # If no documents match, return an empty list
        if not filtered_ids:
            logger.debug("No documents match the provided metadata filters.")
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        if not query_embedding:
            # if no query is being passed then return the nodes which satisfies the passed filters
            nodes, ids = [], []
            for idx, metadata in id_to_metadata.items():
                nodes.append(self.create_node(metadata))
                ids.append(idx)
            return VectorStoreQueryResult(nodes=nodes, similarities=[0] * len(ids), ids=ids)

        # Convert query embedding to the correct format
        query_embedding_np = np.array([query_embedding], dtype="float32").astype("float32")
        # query_embedding_np = query_embedding.reshape(1, -1)  # Reshape to (1, dimension)

        # Perform the search
        distances, indices = self.load_and_search(query_embedding_np, top_k, filtered_ids=filtered_ids)

        # Step 4: Retrieve results and map back to metadata
        nodes = []
        similarities = []
        ids = []

        # Ensure no zero distances to avoid division by zero
        adjusted_distances = np.maximum(distances[0], 1e-9)  # Avoid zero distances
        # Invert the distances (shorter distances mean higher similarity)
        inverse_distances = 1.0 / adjusted_distances
        # Normalize the inverted distances to a 0 to 1 range
        probabilities = inverse_distances / inverse_distances.sum()

        for i, idx in enumerate(indices[0]):
            if idx != -1:  # Ensure that we are not referencing an invalid index
                metadata = id_to_metadata.get(idx) or id_to_metadata.get(str(idx))
                if not metadata:
                    continue
                nodes.append(self.create_node(metadata))  # Use metadata as nodes
                similarities.append(
                    probabilities[i]
                )  # shorted distances means that the node is so similar to the asked question
                ids.append(idx)  # Use FAISS doc_ids as ids

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def _persist_internal(self):
        """
        Internal persist without file lock - caller must hold write lock.
        Uses atomic write (write to temp file, then rename) to prevent corruption.
        """
        logger.info(f"Saving FAISS index to {self._index_file}")
        
        # Atomic write for FAISS index: write to temp file, then rename
        temp_index_file = self._index_file + ".tmp"
        faiss.write_index(self._faiss_index, temp_index_file)
        os.replace(temp_index_file, self._index_file)

        logger.info(f"Saving metadata to {self._metadata_file}")
        # Atomic write for metadata: write to temp file, then rename
        temp_metadata_file = self._metadata_file + ".tmp"
        with open(temp_metadata_file, "wb") as f:
            pickle.dump(self._metadata_store, f)
        os.replace(temp_metadata_file, self._metadata_file)

    def persist(self, *args, **kwargs):
        """
        Persist the FAISS index and metadata to disk with file-based locking.
        Uses atomic write (write to temp file, then rename) to prevent corruption.
        """
        with self._file_lock.write_lock():
            self._persist_internal()

    @classmethod
    def class_name(cls: type["FaissIQVectorStore"]) -> str:
        return "FaissIQVectorStore"
