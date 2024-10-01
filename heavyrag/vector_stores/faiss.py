import os
import pickle
from collections.abc import Sequence
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


class FaissIQVectorStore(FaissVectorStore):
    """
    Faiss vector store wrapper class.
    """

    _persis_dir: str = PrivateAttr()
    _metadata_store: dict = PrivateAttr()
    _index_file: str = PrivateAttr()
    _metadata_file: str = PrivateAttr()
    # _faiss_index = PrivateAttr()  # which lets you map custom ids to vectors

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
        # self.dimension = dimension
        self._persis_dir = persist_dir
        self._metadata_store: dict = {}

        self._index_file = os.path.join(persist_dir, index_file)
        self._metadata_file = os.path.join(persist_dir, metadata_file)

        os.makedirs(persist_dir, exist_ok=True)  # create the persist folder

        # Check if the FAISS index file exists
        if os.path.exists(self._index_file):
            logger.info(f"Loading FAISS index from {self._index_file}")
            index = faiss.read_index(self._index_file)
            self.load_metadata()
        else:
            logger.info(f"Creating new FAISS index with dimension {dimension}")
            index = faiss.IndexIDMap((faiss.IndexFlatL2(dimension)))  # Create new FAISS index

        # self._faiss_index =
        print(type(index))
        super().__init__(faiss_index=index)

    def load_metadata(self):
        """
        Load the metadata from the metadata file.
        """
        if os.path.exists(self._metadata_file):
            logger.info(f"Loading metadata from {self._metadata_file}")
            with open(self._metadata_file, "rb") as f:
                self._metadata_store = pickle.load(f)
        else:
            logger.info("No metadata file found, initializing empty metadata store.")
            self._metadata_store = {}

    def remove(self, ids: list[int] | None = None, node_ids: list[str] | None = None) -> None:
        """
        Remove vectors from the FAISS index as well as from metadata dump using their IDs.
        Supports both node_ids(str) as well as the vector store index ids (int).
        """
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

        self.persist()

    def add(
        self,
        nodes: Sequence[BaseNode],
        **add_kwargs: Any,
    ) -> list[int]:
        """Add nodes to index.

        NOTE: in the Faiss vector store, we do not store text in Faiss.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        """
        new_ids = []
        for node in nodes:
            text_embedding = node.get_embedding()
            text_embedding_np = np.array(text_embedding, dtype="float32").reshape(1, -1)
            new_id = uuid4_hex_to_int64(node.node_id)
            self._faiss_index.add_with_ids(text_embedding_np, np.array([new_id], dtype=np.int64))
            new_ids.append(new_id)
            self._metadata_store[new_id] = {"text": node.get_content(), "metadata": node.metadata}
            # add metadata to the metadatastore with the corresponding doc_id as key
        self.persist()  # do persist to disk after node's addition
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

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """
        Perform a FAISS similarity search with optional metadata filtering using complex MetadataFilters.

        Args:
            query: VectorStoreQuery object that includes query_embedding, and_filters, or_filters, and top_k.
            kwargs: Additional keyword arguments.

        Returns:
            VectorStoreQueryResult: The result object containing nodes, similarities, and ids.
        """
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
            print("No documents match the provided metadata filters.")
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        if not query_embedding:
            # if no query is being passed then return the nodes which satisfies the passed filters
            nodes, ids = [], []
            for idx, metadata in id_to_metadata.items():
                nodes.append(self.create_node(metadata))
                ids.append(idx)
            return VectorStoreQueryResult(nodes=nodes, similarities=[0] * len(ids), ids=ids)

        # Step 2: Create FAISS IDSelector for filtering
        id_selector = faiss.IDSelectorBatch(np.array(filtered_ids, dtype=np.int64))

        # Step 3: Perform FAISS similarity search with IDSelector
        faiss_search_params = faiss.SearchParameters()
        faiss_search_params.selector = id_selector  # Apply the ID selector for filtering

        # Convert query embedding to the correct format
        query_embedding_np = np.array([query_embedding], dtype="float32")
        # query_embedding_np = query_embedding.reshape(1, -1)  # Reshape to (1, dimension)

        # Perform the search
        distances, indices = self._faiss_index.search(query_embedding_np, top_k, params=faiss_search_params)

        # Step 4: Retrieve results and map back to metadata
        nodes = []
        similarities = []
        ids = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # Ensure that we are not referencing an invalid index
                metadata = id_to_metadata.get(idx) or id_to_metadata.get(str(idx))
                nodes.append(self.create_node(metadata))  # Use metadata as nodes
                similarities.append(distances[0][i])  # Use distances as similarities
                ids.append(idx)  # Use FAISS doc_ids as ids

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def persist(self, *args, **kwargs):
        """
        Persist the FAISS index and metadata to disk.
        """
        logger.info(f"Saving FAISS index to {self._index_file}")
        faiss.write_index(self._faiss_index, self._index_file)

        logger.info(f"Saving metadata to {self._metadata_file}")
        with open(self._metadata_file, "wb") as f:
            pickle.dump(self._metadata_store, f)

    @classmethod
    def class_name(cls: type["FaissIQVectorStore"]) -> str:
        return "FaissIQVectorStore"
