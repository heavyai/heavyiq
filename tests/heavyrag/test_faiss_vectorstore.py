"""
Tests for FAISS vector store.

Tests the FaissIQVectorStore class and related functionality.
"""

import os
import shutil
import tempfile
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch
from uuid import uuid4

import faiss
import numpy as np
import pytest

from heavyiq.config import HeavyIQConfig


def generate_hex_id() -> str:
    """Generate a valid hex UUID for FAISS node IDs."""
    return uuid4().hex


class TestFaissVectorStore:
    """Tests for FaissIQVectorStore."""

    @pytest.fixture
    def temp_persist_dir(self):
        """Create a temporary directory for FAISS persistence."""
        temp_dir = tempfile.mkdtemp(prefix="faiss_test_")
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_config(self, temp_persist_dir):
        """Create a mock config for testing."""
        config = MagicMock(spec=HeavyIQConfig)
        config.rag_faiss_persist_dir = temp_persist_dir
        config.rag_embed_dimension = 384  # Common embedding dimension
        config.rag_vectordb_type = "faiss"
        return config

    @pytest.fixture
    def faiss_vectorstore(self, mock_config, temp_persist_dir):
        """Create a FaissIQVectorStore instance for testing."""
        with patch("heavyrag.vector_stores.faiss.get_faiss_index") as mock_get_index:
            # Create a real FAISS index for testing
            # Use IndexIDMap to support add_with_ids (IndexFlatL2 alone doesn't support custom IDs)
            dimension = mock_config.rag_embed_dimension
            index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
            mock_get_index.return_value = index

            from heavyrag.vector_stores.faiss import FaissIQVectorStore

            store = FaissIQVectorStore(
                persist_dir=temp_persist_dir,
                dimension=dimension,
            )
            yield store

    def test_init_creates_persist_dir(self, mock_config, temp_persist_dir):
        """Test that initialization creates the persist directory."""
        # Remove the temp dir first
        shutil.rmtree(temp_persist_dir)
        assert not os.path.exists(temp_persist_dir)

        with patch("heavyrag.vector_stores.faiss.get_faiss_index") as mock_get_index:
            dimension = mock_config.rag_embed_dimension
            index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
            mock_get_index.return_value = index

            from heavyrag.vector_stores.faiss import FaissIQVectorStore

            store = FaissIQVectorStore(
                persist_dir=temp_persist_dir,
                dimension=dimension,
            )

            assert os.path.exists(temp_persist_dir)

    def test_add_nodes(self, faiss_vectorstore, mock_config):
        """Test adding nodes to the vector store."""
        from llama_index.core.schema import TextNode

        # Create test nodes with embeddings (using valid hex UUIDs)
        dimension = mock_config.rag_embed_dimension
        node1_id = generate_hex_id()
        node2_id = generate_hex_id()
        nodes = [
            TextNode(
                text="Test document 1",
                id_=node1_id,
                embedding=np.random.rand(dimension).tolist(),
                metadata={"source": "test1"},
            ),
            TextNode(
                text="Test document 2",
                id_=node2_id,
                embedding=np.random.rand(dimension).tolist(),
                metadata={"source": "test2"},
            ),
        ]

        # Add nodes
        result = faiss_vectorstore.add(nodes)

        assert len(result) == 2

    def test_persist_and_load(self, mock_config, temp_persist_dir):
        """Test persisting and loading the index."""
        with patch("heavyrag.vector_stores.faiss.get_faiss_index") as mock_get_index:
            from llama_index.core.schema import TextNode

            dimension = mock_config.rag_embed_dimension
            index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
            mock_get_index.return_value = index

            from heavyrag.vector_stores.faiss import FaissIQVectorStore

            store = FaissIQVectorStore(
                persist_dir=temp_persist_dir,
                dimension=dimension,
            )

            # Add a node
            test_node_id = generate_hex_id()
            node = TextNode(
                text="Test document",
                id_=test_node_id,
                embedding=np.random.rand(dimension).tolist(),
                metadata={"key": "value"},
            )
            store.add([node])

            # Persist
            store.persist()

            # Check files exist
            assert os.path.exists(os.path.join(temp_persist_dir, "faiss_index.bin"))
            assert os.path.exists(os.path.join(temp_persist_dir, "metadata.pkl"))

    def test_delete_nodes(self, faiss_vectorstore, mock_config):
        """Test deleting nodes from the vector store."""
        from llama_index.core.schema import TextNode

        dimension = mock_config.rag_embed_dimension

        # Add nodes first (using valid hex UUIDs)
        delete_node1_id = generate_hex_id()
        delete_node2_id = generate_hex_id()
        nodes = [
            TextNode(
                text="Test document 1",
                id_=delete_node1_id,
                embedding=np.random.rand(dimension).tolist(),
                metadata={"source": "test1"},
            ),
            TextNode(
                text="Test document 2",
                id_=delete_node2_id,
                embedding=np.random.rand(dimension).tolist(),
                metadata={"source": "test2"},
            ),
        ]
        faiss_vectorstore.add(nodes)

        # Delete one node using FAISS remove method
        faiss_vectorstore.remove(node_ids=[delete_node1_id])

        # The node should be removed from metadata
        from heavyrag.utils import uuid4_hex_to_int64
        node1_int_id = uuid4_hex_to_int64(delete_node1_id)
        assert node1_int_id not in faiss_vectorstore._metadata_store

    def test_query(self, faiss_vectorstore, mock_config):
        """Test querying the vector store."""
        from llama_index.core.schema import TextNode
        from llama_index.core.vector_stores.types import VectorStoreQuery

        dimension = mock_config.rag_embed_dimension

        # Add nodes (using valid hex UUIDs)
        python_node_id = generate_hex_id()
        js_node_id = generate_hex_id()
        nodes = [
            TextNode(
                text="Python is a programming language",
                id_=python_node_id,
                embedding=np.random.rand(dimension).tolist(),
                metadata={"topic": "programming"},
            ),
            TextNode(
                text="JavaScript is also a programming language",
                id_=js_node_id,
                embedding=np.random.rand(dimension).tolist(),
                metadata={"topic": "programming"},
            ),
        ]
        faiss_vectorstore.add(nodes)

        # Query
        query_embedding = np.random.rand(dimension).tolist()
        query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=2,
        )

        result = faiss_vectorstore.query(query)

        assert result is not None
        assert hasattr(result, "nodes")
        assert hasattr(result, "similarities")

    def test_reset(self, faiss_vectorstore, mock_config):
        """Test resetting the vector store."""
        from llama_index.core.schema import TextNode

        dimension = mock_config.rag_embed_dimension

        # Add a node (using valid hex UUID)
        reset_node_id = generate_hex_id()
        node = TextNode(
            text="Test document",
            id_=reset_node_id,
            embedding=np.random.rand(dimension).tolist(),
            metadata={"key": "value"},
        )
        faiss_vectorstore.add([node])

        # Verify node exists
        assert len(faiss_vectorstore._metadata_store) > 0

        # Reset
        faiss_vectorstore.reset()

        # Verify cleared
        assert len(faiss_vectorstore._metadata_store) == 0

    def test_get_ids(self, faiss_vectorstore, mock_config):
        """Test getting all IDs from the vector store."""
        from llama_index.core.schema import TextNode

        dimension = mock_config.rag_embed_dimension

        # Add nodes (using valid hex UUIDs)
        id1 = generate_hex_id()
        id2 = generate_hex_id()
        nodes = [
            TextNode(
                text="Test document 1",
                id_=id1,
                embedding=np.random.rand(dimension).tolist(),
            ),
            TextNode(
                text="Test document 2",
                id_=id2,
                embedding=np.random.rand(dimension).tolist(),
            ),
        ]
        faiss_vectorstore.add(nodes)

        ids = faiss_vectorstore.get_ids()

        assert len(ids) >= 2

    def test_class_name(self, faiss_vectorstore):
        """Test class_name class method."""
        assert faiss_vectorstore.class_name() == "FaissIQVectorStore"


class TestFaissIndex:
    """Tests for FAISS index management functions."""

    @pytest.fixture
    def temp_persist_dir(self):
        """Create a temporary directory for FAISS persistence."""
        temp_dir = tempfile.mkdtemp(prefix="faiss_index_test_")
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_get_faiss_index_creates_new(self, temp_persist_dir):
        """Test get_faiss_index creates a new index when none exists."""
        # Reset global faiss_index
        import heavyrag.vector_stores.faiss as faiss_module

        original_index = faiss_module.faiss_index
        faiss_module.faiss_index = None

        try:
            from heavyrag.vector_stores.faiss import get_faiss_index

            index = get_faiss_index(persist_dir=temp_persist_dir, dimension=128)

            assert index is not None
        finally:
            faiss_module.faiss_index = original_index

    def test_get_faiss_index_loads_existing(self, temp_persist_dir):
        """Test get_faiss_index loads an existing index from disk."""
        import heavyrag.vector_stores.faiss as faiss_module

        original_index = faiss_module.faiss_index
        faiss_module.faiss_index = None

        try:
            # Create and save an index
            dimension = 128
            index = faiss.IndexFlatL2(dimension)

            # Add some vectors
            vectors = np.random.rand(5, dimension).astype("float32")
            index.add(vectors)

            index_path = os.path.join(temp_persist_dir, "faiss_index.bin")
            faiss.write_index(index, index_path)

            # Now load it
            from heavyrag.vector_stores.faiss import get_faiss_index

            loaded_index = get_faiss_index(persist_dir=temp_persist_dir, dimension=dimension)

            assert loaded_index is not None
            assert loaded_index.ntotal == 5  # Should have 5 vectors
        finally:
            faiss_module.faiss_index = original_index

    def test_get_faiss_index_cached(self, temp_persist_dir):
        """Test get_faiss_index returns cached index on subsequent calls."""
        import heavyrag.vector_stores.faiss as faiss_module

        original_index = faiss_module.faiss_index
        faiss_module.faiss_index = None

        try:
            from heavyrag.vector_stores.faiss import get_faiss_index

            index1 = get_faiss_index(persist_dir=temp_persist_dir, dimension=128)
            index2 = get_faiss_index(persist_dir=temp_persist_dir, dimension=128)

            # Should be the same cached instance
            assert index1 is index2
        finally:
            faiss_module.faiss_index = original_index


class TestFaissController:
    """Tests for the FAISS controller."""

    @pytest.fixture
    def temp_persist_dir(self):
        """Create a temporary directory for FAISS persistence."""
        temp_dir = tempfile.mkdtemp(prefix="faiss_controller_test_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_config(self, temp_persist_dir):
        """Create a mock config for testing."""
        config = MagicMock()
        config.rag_faiss_persist_dir = temp_persist_dir
        config.rag_embed_dimension = 384
        config.rag_vectordb_type = "faiss"
        config.rag_embed_server_base = "http://localhost:8080"
        return config

    def test_faiss_controller_persist_dir(self, mock_config):
        """Test FaissController persist_dir property."""
        with patch("heavyrag.controller.CONFIG", mock_config):
            from heavyrag.controller import FaissController

            controller = FaissController()

            assert controller.persist_dir == mock_config.rag_faiss_persist_dir

    def test_faiss_controller_get_vectorstore_caches(self, mock_config, temp_persist_dir):
        """Test FaissController caches the vectorstore."""
        with patch("heavyrag.controller.CONFIG", mock_config), patch(
            "heavyrag.controller.EMBED_MODEL", MagicMock()
        ), patch("heavyrag.vector_stores.faiss.get_faiss_index") as mock_get_index:
            # Setup mock - use IndexIDMap to support add_with_ids
            index = faiss.IndexIDMap(faiss.IndexFlatL2(384))
            mock_get_index.return_value = index

            from heavyrag.controller import FaissController

            controller = FaissController()
            controller._cached_vector_store = None

            # First call should create
            store1 = controller.get_vectorstore()

            # Second call should return cached
            store2 = controller.get_vectorstore()

            assert store1 is store2


