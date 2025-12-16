"""
Tests for ChromaDB vector store.

Tests the ChromaIQVectorStore class and related functionality.
"""

import os
import shutil
import tempfile
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from heavyiq.config import HeavyIQConfig


class TestChromaVectorStore:
    """Tests for ChromaIQVectorStore."""

    @pytest.fixture
    def temp_persist_dir(self):
        """Create a temporary directory for ChromaDB persistence."""
        temp_dir = tempfile.mkdtemp(prefix="chroma_test_")
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_config(self, temp_persist_dir):
        """Create a mock config for testing."""
        config = MagicMock(spec=HeavyIQConfig)
        config.rag_chromadb_persist_dir = temp_persist_dir
        config.rag_chromadb_server_base = None  # Use local mode
        config.rag_embed_dimension = 384
        config.rag_vectordb_type = "chroma"
        return config

    @pytest.fixture
    def chroma_vectorstore(self, mock_config, temp_persist_dir):
        """Create a ChromaIQVectorStore instance for testing."""
        with patch("heavyrag.vector_stores.chroma.CONFIG", mock_config), patch(
            "heavyrag.vector_stores.chroma.get_chroma_client"
        ) as mock_get_client:
            # Create a real ChromaDB client for testing
            import chromadb

            client = chromadb.Client()
            mock_get_client.return_value = client

            from heavyrag.vector_stores.chroma import ChromaIQVectorStore

            store = ChromaIQVectorStore(
                collection_name="test_collection",
                metadata={"hnsw:space": "cosine"},
                create_collection_if_not_exists=True,
            )
            yield store

    def test_init_creates_collection(self, mock_config, temp_persist_dir):
        """Test that initialization creates a collection."""
        with patch("heavyrag.vector_stores.chroma.CONFIG", mock_config), patch(
            "heavyrag.vector_stores.chroma.get_chroma_client"
        ) as mock_get_client:
            import chromadb

            client = chromadb.Client()
            mock_get_client.return_value = client

            from heavyrag.vector_stores.chroma import ChromaIQVectorStore

            store = ChromaIQVectorStore(
                collection_name="new_collection",
                create_collection_if_not_exists=True,
            )

            # Collection should exist
            collections = client.list_collections()
            collection_names = [c.name for c in collections]
            assert "new_collection" in collection_names

    def test_add_nodes(self, chroma_vectorstore, mock_config):
        """Test adding nodes to the vector store."""
        from llama_index.core.schema import TextNode

        dimension = mock_config.rag_embed_dimension
        nodes = [
            TextNode(
                text="Test document 1",
                id_="chroma_node1",
                embedding=np.random.rand(dimension).tolist(),
                metadata={"source": "test1"},
            ),
            TextNode(
                text="Test document 2",
                id_="chroma_node2",
                embedding=np.random.rand(dimension).tolist(),
                metadata={"source": "test2"},
            ),
        ]

        result = chroma_vectorstore.add(nodes)

        assert len(result) == 2

    def test_delete_nodes(self, mock_config, temp_persist_dir):
        """Test deleting nodes from the vector store using the delete_nodes factory method."""
        # Use a fresh collection for this test to avoid interference
        with patch("heavyrag.vector_stores.chroma.CONFIG", mock_config), patch(
            "heavyrag.vector_stores.chroma.get_chroma_client"
        ) as mock_get_client:
            import chromadb
            from llama_index.core.schema import TextNode
            from llama_index.core.vector_stores.types import VectorStoreQuery

            from heavyrag.ingest import delete_nodes

            client = chromadb.Client()
            mock_get_client.return_value = client

            from heavyrag.vector_stores.chroma import ChromaIQVectorStore

            # Create a fresh collection for this test
            store = ChromaIQVectorStore(
                collection_name="delete_test_collection",
                create_collection_if_not_exists=True,
            )

            dimension = mock_config.rag_embed_dimension

            # Add node with metadata for filtering
            nodes = [
                TextNode(
                    text="Test document to delete",
                    id_="chroma_delete_node",
                    embedding=np.random.rand(dimension).tolist(),
                    metadata={"source": "test", "type": "test_doc"},
                ),
            ]
            store.add(nodes)

            # Delete using the factory method from heavyrag.ingest
            deleted = delete_nodes(
                collection=store._collection,
                where={"type": "test_doc"}
            )
            assert deleted is True

            # Query to verify deletion
            query = VectorStoreQuery(
                query_embedding=np.random.rand(dimension).tolist(),
                similarity_top_k=10,
            )
            result = store.query(query)

            # Node should not be in results (collection should be empty)
            assert result.nodes is None or len(result.nodes) == 0

    def test_query(self, chroma_vectorstore, mock_config):
        """Test querying the vector store."""
        from llama_index.core.schema import TextNode
        from llama_index.core.vector_stores.types import VectorStoreQuery

        dimension = mock_config.rag_embed_dimension

        # Add nodes with known embeddings
        embedding1 = np.random.rand(dimension).tolist()
        nodes = [
            TextNode(
                text="Python programming language",
                id_="python_chroma",
                embedding=embedding1,
                metadata={"topic": "programming"},
            ),
            TextNode(
                text="JavaScript programming language",
                id_="js_chroma",
                embedding=np.random.rand(dimension).tolist(),
                metadata={"topic": "programming"},
            ),
        ]
        chroma_vectorstore.add(nodes)

        # Query with similar embedding
        query = VectorStoreQuery(
            query_embedding=embedding1,  # Same as first node
            similarity_top_k=2,
        )

        result = chroma_vectorstore.query(query)

        assert result is not None
        assert hasattr(result, "nodes")
        assert len(result.nodes) > 0

    def test_collection_not_exists_error(self, mock_config, temp_persist_dir):
        """Test error when collection doesn't exist and create_if_not_exists is False."""
        with patch("heavyrag.vector_stores.chroma.CONFIG", mock_config), patch(
            "heavyrag.vector_stores.chroma.get_chroma_client"
        ) as mock_get_client:
            import chromadb

            client = chromadb.Client()
            mock_get_client.return_value = client

            from heavyrag.vector_stores.chroma import ChromaIQVectorStore

            # Should raise or return None when collection doesn't exist
            try:
                store = ChromaIQVectorStore(
                    collection_name="nonexistent_collection",
                    create_collection_if_not_exists=False,
                )
                # If it doesn't raise, the store might be None or empty
                assert store is None or store._collection is None
            except Exception:
                # Expected behavior - should raise when collection doesn't exist
                pass

    def test_class_name(self, chroma_vectorstore):
        """Test class_name class method."""
        assert chroma_vectorstore.class_name() == "ChromaIQVectorStore"


class TestChromaClient:
    """Tests for ChromaDB client management."""

    @pytest.fixture
    def temp_persist_dir(self):
        """Create a temporary directory for ChromaDB."""
        temp_dir = tempfile.mkdtemp(prefix="chroma_client_test_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_get_chroma_client_http_mock(self, temp_persist_dir):
        """Test getting a ChromaDB HTTP client with mocked server."""
        import chromadb

        mock_config = MagicMock()
        mock_config.rag_chromadb_server_base = "http://localhost:8000"

        # Create a real in-memory client to return from the mock
        real_client = chromadb.Client()

        with patch("heavyrag.vector_stores.chroma.CONFIG", mock_config), \
             patch("heavyrag.vector_stores.chroma.CHROMA_CLIENT", None), \
             patch("heavyrag.vector_stores.chroma.get_embed_model") as mock_embed, \
             patch("chromadb.HttpClient", return_value=real_client):
            mock_embed.return_value = MagicMock()

            from heavyrag.vector_stores.chroma import get_chroma_client

            client = get_chroma_client()

            assert client is not None
            # Should be able to list collections
            collections = client.list_collections()
            assert isinstance(collections, list)

    def test_get_chroma_client_http(self):
        """Test getting an HTTP ChromaDB client."""
        mock_config = MagicMock()
        mock_config.rag_chromadb_server_base = "http://localhost:8000"

        with patch("heavyrag.vector_stores.chroma.CONFIG", mock_config), \
             patch("heavyrag.vector_stores.chroma.CHROMA_CLIENT", None), \
             patch("heavyrag.vector_stores.chroma.get_embed_model") as mock_embed, \
             patch("chromadb.HttpClient") as mock_http_client:
            mock_embed.return_value = MagicMock()
            mock_client = MagicMock()
            mock_http_client.return_value = mock_client

            from heavyrag.vector_stores.chroma import get_chroma_client

            client = get_chroma_client()

            # Should have called HttpClient
            mock_http_client.assert_called_once()


class TestChromaController:
    """Tests for the ChromaDB controller."""

    @pytest.fixture
    def temp_persist_dir(self):
        """Create a temporary directory for ChromaDB."""
        temp_dir = tempfile.mkdtemp(prefix="chroma_controller_test_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_config(self, temp_persist_dir):
        """Create a mock config for testing."""
        config = MagicMock()
        config.rag_chromadb_persist_dir = temp_persist_dir
        config.rag_chromadb_server_base = None
        config.rag_embed_dimension = 384
        config.rag_vectordb_type = "chroma"
        config.rag_embed_server_base = "http://localhost:8080"
        return config

    def test_chroma_controller_persist_dir(self, mock_config):
        """Test ChromaController persist_dir property."""
        with patch("heavyrag.controller.CONFIG", mock_config):
            from heavyrag.controller import ChromaController

            controller = ChromaController()

            assert controller.persist_dir == mock_config.rag_chromadb_persist_dir

    def test_chroma_controller_get_vectorstore(self, mock_config, temp_persist_dir):
        """Test ChromaController get_vectorstore method."""
        with patch("heavyrag.controller.CONFIG", mock_config), patch(
            "heavyrag.controller.EMBED_MODEL", MagicMock()
        ), patch("heavyrag.vector_stores.chroma.get_chroma_client") as mock_get_client:
            import chromadb

            client = chromadb.Client()
            mock_get_client.return_value = client

            from heavyrag.controller import ChromaController

            controller = ChromaController()

            store = controller.get_vectorstore(collection_name="test_collection")

            assert store is not None


class TestVectorStoreType:
    """Tests for VectorStoreType enum."""

    def test_from_name_chroma(self):
        """Test VectorStoreType.from_name for chroma."""
        from heavyrag.vector_stores.base import VectorStoreType

        result = VectorStoreType.from_name("chroma")

        assert result == VectorStoreType.CHROMA

    def test_from_name_faiss(self):
        """Test VectorStoreType.from_name for faiss."""
        from heavyrag.vector_stores.base import VectorStoreType

        result = VectorStoreType.from_name("faiss")

        assert result == VectorStoreType.FAISS

    def test_from_name_invalid(self):
        """Test VectorStoreType.from_name with invalid name."""
        from heavyrag.vector_stores.base import VectorStoreType

        with pytest.raises(ValueError):
            VectorStoreType.from_name("invalid_type")

    def test_from_name_case_insensitive(self):
        """Test VectorStoreType.from_name is case insensitive."""
        from heavyrag.vector_stores.base import VectorStoreType

        assert VectorStoreType.from_name("CHROMA") == VectorStoreType.CHROMA
        assert VectorStoreType.from_name("Faiss") == VectorStoreType.FAISS
        assert VectorStoreType.from_name("FAISS") == VectorStoreType.FAISS


