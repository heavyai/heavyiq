# Test RAG controller module
# Unit tests use mocks, integration tests require actual services
"""
Tests for heavyrag/controller.py

Unit tests: Test individual methods with mocks (no external dependencies)
Integration tests: Require actual services (ChromaDB, embedding server, HeavyDB)
"""
import os
import shutil
import tempfile
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import faiss
import numpy as np
import pytest

if TYPE_CHECKING:
    from heavyiq.langchain.heavydb import HeavyDB
    from heavyrag.controller import BaseController


def get_uuid() -> str:
    """Generate a hex UUID."""
    return uuid4().hex


# =============================================================================
# UNIT TESTS - Use mocks, no external dependencies
# =============================================================================


class TestGetController:
    """Unit tests for get_controller factory function."""

    def test_get_controller_chroma(self):
        """Test get_controller returns ChromaController for 'chroma'."""
        with patch("heavyrag.controller.CONFIG") as mock_config:
            mock_config.rag_vectordb_type = "chroma"
            mock_config.rag_chromadb_persist_dir = "/tmp/test"
            from heavyrag.controller import ChromaController, get_controller

            controller = get_controller("chroma")
            assert isinstance(controller, ChromaController)

    def test_get_controller_faiss(self):
        """Test get_controller returns FaissController for 'faiss'."""
        with patch("heavyrag.controller.CONFIG") as mock_config:
            mock_config.rag_vectordb_type = "faiss"
            mock_config.rag_faiss_persist_dir = "/tmp/test"
            from heavyrag.controller import FaissController, get_controller

            controller = get_controller("faiss")
            assert isinstance(controller, FaissController)

    def test_get_controller_unknown_raises(self):
        """Test get_controller raises ValueError for unknown type."""
        from heavyrag.controller import get_controller

        with pytest.raises(ValueError, match="Unknown controller type"):
            get_controller("unknown")

    def test_get_controller_case_insensitive(self):
        """Test get_controller is case insensitive."""
        with patch("heavyrag.controller.CONFIG") as mock_config:
            mock_config.rag_vectordb_type = "faiss"
            mock_config.rag_faiss_persist_dir = "/tmp/test"
            from heavyrag.controller import FaissController, get_controller

            controller = get_controller("FAISS")
            assert isinstance(controller, FaissController)


class TestFactAbstract:
    """Unit tests for FactAbstract methods."""

    @pytest.fixture
    def mock_controller(self):
        """Create a mock controller with FactAbstract methods."""
        with patch("heavyrag.controller.CONFIG") as mock_config, patch(
            "heavyrag.controller.EMBED_MODEL", MagicMock()
        ):
            mock_config.rag_faiss_persist_dir = "/tmp/test"
            mock_config.rag_embed_dimension = 384
            from heavyrag.controller import FaissController

            controller = FaissController()
            return controller

    def test_fact_metadata(self, mock_controller):
        """Test _fact_metadata generates correct metadata dict."""
        dbname = "test_db"
        fact_id = get_uuid()

        metadata = mock_controller._fact_metadata(dbname=dbname, fact_id=fact_id)

        assert metadata["dbname"] == dbname
        assert metadata["type"] == "facts"
        assert metadata["id"] == fact_id

    def test_fact_metadata_with_extra_kwargs(self, mock_controller):
        """Test _fact_metadata includes extra kwargs."""
        dbname = "test_db"
        fact_id = get_uuid()

        metadata = mock_controller._fact_metadata(
            dbname=dbname, fact_id=fact_id, custom_field="value"
        )

        assert metadata["custom_field"] == "value"

    def test_load_facts(self, mock_controller):
        """Test _load_facts creates TextNodes from fact tuples."""
        dbname = "test_db"
        facts = [
            (get_uuid(), "Fact 1 content"),
            (get_uuid(), "Fact 2 content"),
        ]

        nodes = mock_controller._load_facts(facts=facts, dbname=dbname)

        assert len(nodes) == 2
        assert nodes[0].text == "Fact 1 content"
        assert nodes[1].text == "Fact 2 content"
        assert nodes[0].metadata["dbname"] == dbname
        assert nodes[0].metadata["type"] == "facts"


class TestBaseControllerMethods:
    """Unit tests for BaseController shared methods."""

    @pytest.fixture
    def temp_persist_dir(self):
        """Create a temporary directory for persistence."""
        temp_dir = tempfile.mkdtemp(prefix="controller_test_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_persist_dir_property(self, temp_persist_dir):
        """Test persist_dir returns correct path."""
        with patch("heavyrag.controller.CONFIG") as mock_config:
            mock_config.rag_faiss_persist_dir = temp_persist_dir

            from heavyrag.controller import FaissController

            controller = FaissController()
            assert controller.persist_dir == temp_persist_dir

    def test_create_docstore(self):
        """Test create_docstore returns SimpleDocumentStore."""
        from llama_index.core.storage.docstore import SimpleDocumentStore

        with patch("heavyrag.controller.CONFIG") as mock_config:
            mock_config.rag_faiss_persist_dir = "/tmp/test"

            from heavyrag.controller import FaissController

            controller = FaissController()
            docstore = controller.create_docstore()
            assert isinstance(docstore, SimpleDocumentStore)

    def test_get_or_create_docstore_creates_new(self, temp_persist_dir):
        """Test get_or_create_docstore creates new when none exists."""
        from llama_index.core.storage.docstore import SimpleDocumentStore

        with patch("heavyrag.controller.CONFIG") as mock_config:
            mock_config.rag_faiss_persist_dir = temp_persist_dir

            from heavyrag.controller import FaissController

            controller = FaissController()
            docstore = controller.get_or_create_docstore()
            assert isinstance(docstore, SimpleDocumentStore)


class TestFaissControllerUnit:
    """Unit tests for FaissController specific methods."""

    @pytest.fixture
    def temp_persist_dir(self):
        """Create a temporary directory for FAISS persistence."""
        temp_dir = tempfile.mkdtemp(prefix="faiss_controller_test_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def faiss_controller(self, temp_persist_dir):
        """Create a FaissController with mocked dependencies."""
        with patch("heavyrag.controller.CONFIG") as mock_config, patch(
            "heavyrag.controller.EMBED_MODEL", MagicMock()
        ), patch("heavyrag.vector_stores.faiss.get_faiss_index") as mock_get_index:
            mock_config.rag_faiss_persist_dir = temp_persist_dir
            mock_config.rag_embed_dimension = 384

            index = faiss.IndexIDMap(faiss.IndexFlatL2(384))
            mock_get_index.return_value = index

            from heavyrag.controller import FaissController

            controller = FaissController()
            controller._cached_vector_store = None
            yield controller

    def test_get_vectorstore_caches(self, faiss_controller):
        """Test get_vectorstore caches the vector store."""
        store1 = faiss_controller.get_vectorstore()
        store2 = faiss_controller.get_vectorstore()
        assert store1 is store2

    def test_get_vectorstore_returns_none_without_embed_model(self, temp_persist_dir):
        """Test get_vectorstore returns None when EMBED_MODEL is None."""
        with patch("heavyrag.controller.CONFIG") as mock_config, patch(
            "heavyrag.controller.EMBED_MODEL", None
        ):
            mock_config.rag_faiss_persist_dir = temp_persist_dir
            mock_config.rag_embed_dimension = 384

            from heavyrag.controller import FaissController

            controller = FaissController()
            controller._cached_vector_store = None

            store = controller.get_vectorstore()
            assert store is None


class TestChromaControllerUnit:
    """Unit tests for ChromaController specific methods."""

    @pytest.fixture
    def temp_persist_dir(self):
        """Create a temporary directory for ChromaDB persistence."""
        temp_dir = tempfile.mkdtemp(prefix="chroma_controller_test_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_max_batch_size(self):
        """Test MAX_BATCH_SIZE constant is set correctly."""
        from heavyrag.controller import ChromaController

        assert ChromaController.MAX_BATCH_SIZE == 166

    def test_get_vectorstore_returns_none_without_embed_model(self, temp_persist_dir):
        """Test get_vectorstore returns None when EMBED_MODEL is None."""
        with patch("heavyrag.controller.CONFIG") as mock_config, patch(
            "heavyrag.controller.EMBED_MODEL", None
        ):
            mock_config.rag_chromadb_persist_dir = temp_persist_dir

            from heavyrag.controller import ChromaController

            controller = ChromaController()

            store = controller.get_vectorstore(collection_name="test")
            assert store is None


class TestTableAbstract:
    """Unit tests for TableAbstract methods."""

    @pytest.fixture
    def mock_controller(self):
        """Create a mock controller with TableAbstract methods."""
        with patch("heavyrag.controller.CONFIG") as mock_config, patch(
            "heavyrag.controller.EMBED_MODEL", MagicMock()
        ):
            mock_config.rag_faiss_persist_dir = "/tmp/test"
            mock_config.rag_embed_dimension = 384
            from heavyrag.controller import FaissController

            return FaissController()

    @pytest.mark.anyio
    async def test_docs_to_nodes(self, mock_controller):
        """Test _docs_to_nodes transforms documents to nodes."""
        with patch("heavyrag.controller.atransform") as mock_transform:
            from llama_index.core.schema import Document, TextNode

            mock_nodes = [TextNode(text="chunk1"), TextNode(text="chunk2")]
            mock_transform.return_value = mock_nodes

            docs = [Document(text="Test document content")]
            result = await mock_controller._docs_to_nodes(docs)

            mock_transform.assert_called_once_with(documents=docs)
            assert result == mock_nodes

    @pytest.mark.anyio
    async def test_load_table(self, mock_controller):
        """Test _load_table calls aload_table."""
        with patch("heavyrag.controller.aload_table") as mock_aload:
            from llama_index.core.schema import Document

            mock_docs = [Document(text="Table schema")]
            mock_aload.return_value = mock_docs

            result = await mock_controller._load_table("arg1", kwarg1="value")

            mock_aload.assert_called_once_with("arg1", kwarg1="value")
            assert result == mock_docs

    @pytest.mark.anyio
    async def test_load_tables(self, mock_controller):
        """Test _load_tables calls aload_tables."""
        with patch("heavyrag.controller.aload_tables") as mock_aload:
            from llama_index.core.schema import Document

            mock_docs = [Document(text="Table1"), Document(text="Table2")]
            mock_aload.return_value = mock_docs

            result = await mock_controller._load_tables("arg1", kwarg1="value")

            mock_aload.assert_called_once_with("arg1", kwarg1="value")
            assert result == mock_docs


# =============================================================================
# INTEGRATION TESTS - Require actual services
# =============================================================================

# Mark all integration tests with a custom marker
pytestmark_integration = pytest.mark.integration


@pytest.fixture(scope="function")
def facts_data() -> list:
    """Generate test facts data."""
    return [
        (
            get_uuid(),
            "Archaeologists have found pots of honey in ancient Egyptian tombs that are over 3,000 years old and still perfectly edible.",
        ),
        (
            get_uuid(),
            "Botanically, bananas qualify as berries because they develop from a flower with a single ovary and contain multiple seeds.",
        ),
        (
            get_uuid(),
            "Octopuses has two of the hearts pump blood to the gills, while the third pumps it to the rest of the body.",
        ),
    ]


@pytest.mark.integration
@pytest.mark.anyio
async def test_chroma_vectorstore_passes_facts_insert_and_list(
    chroma_controller_patch: "BaseController", facts_data: list
):
    """
    Integration test: insert/listing facts on chroma vectorstore.
    Requires: ChromaDB server, embedding server
    """
    first_dbname, second_dbname = "heavyai", "heavyai_test"
    *first_set, second_set = facts_data
    await chroma_controller_patch.insert_fact_nodes(facts=list(first_set), dbname=first_dbname)
    await chroma_controller_patch.insert_fact_nodes(facts=[second_set], dbname=second_dbname)
    first_nodes = await chroma_controller_patch.list_fact_nodes(first_dbname)
    second_nodes = await chroma_controller_patch.list_fact_nodes(second_dbname)
    assert all(i.get_content() in [k[1] for k in first_set] for i in first_nodes)
    assert all(i.get_content() in [second_set[1]] for i in second_nodes)


@pytest.mark.integration
@pytest.mark.anyio
async def test_chroma_vectorstore_passes_facts_delete(
    chroma_controller_patch: "BaseController", facts_data: list
):
    """
    Integration test: delete facts on chroma vectorstore.
    Requires: ChromaDB server, embedding server
    """
    first_dbname, second_dbname = "heavyai", "heavyai_test"
    *first_set, second_set = facts_data

    first_nodes = await chroma_controller_patch.list_fact_nodes(first_dbname)
    second_nodes = await chroma_controller_patch.list_fact_nodes(second_dbname)
    assert not (first_nodes)
    assert not (second_nodes)
    # insert nodes
    await chroma_controller_patch.insert_fact_nodes(facts=list(first_set), dbname=first_dbname)
    await chroma_controller_patch.insert_fact_nodes(facts=[second_set], dbname=second_dbname)
    # check nodes after insertion
    first_nodes = await chroma_controller_patch.list_fact_nodes(first_dbname)
    second_nodes = await chroma_controller_patch.list_fact_nodes(second_dbname)
    assert first_nodes
    assert second_nodes
    # single node deletion
    first_fact_id = first_nodes[0].node_id
    await chroma_controller_patch.delete_fact_nodes(dbname=first_dbname, fact_ids=[first_fact_id])
    first_nodes = await chroma_controller_patch.list_fact_nodes(first_dbname)
    assert len(first_nodes) == 1
    # delete all nodes
    await chroma_controller_patch.delete_fact_nodes(dbname=second_dbname)
    second_nodes = await chroma_controller_patch.list_fact_nodes(second_dbname)
    assert not (second_nodes)


@pytest.mark.integration
@pytest.mark.anyio
async def test_faiss_vectorstore_passes_facts_insert_and_list(
    faiss_controller_patch: "BaseController", facts_data: list
):
    """
    Integration test: insert/listing facts on faiss vectorstore.
    Requires: embedding server
    """
    first_dbname, second_dbname = "heavyai", "heavyai_test"
    *first_set, second_set = facts_data
    await faiss_controller_patch.insert_fact_nodes(facts=list(first_set), dbname=first_dbname)
    await faiss_controller_patch.insert_fact_nodes(facts=[second_set], dbname=second_dbname)
    first_nodes = await faiss_controller_patch.list_fact_nodes(first_dbname)
    second_nodes = await faiss_controller_patch.list_fact_nodes(second_dbname)
    assert all(i.get_content() in [k[1] for k in first_set] for i in first_nodes)
    assert all(i.get_content() in [second_set[1]] for i in second_nodes)


@pytest.mark.integration
@pytest.mark.anyio
async def test_faiss_vectorstore_passes_facts_delete(
    faiss_controller_patch: "BaseController", facts_data: list
):
    """
    Integration test: delete facts on faiss vectorstore.
    Requires: embedding server
    """
    first_dbname, second_dbname = "heavyai", "heavyai_test"
    *first_set, second_set = facts_data

    vs = faiss_controller_patch.get_vectorstore()
    vs.reset()

    first_nodes = await faiss_controller_patch.list_fact_nodes(first_dbname)
    second_nodes = await faiss_controller_patch.list_fact_nodes(second_dbname)
    assert not (first_nodes)
    assert not (second_nodes)
    # insert nodes
    await faiss_controller_patch.insert_fact_nodes(facts=list(first_set), dbname=first_dbname)
    await faiss_controller_patch.insert_fact_nodes(facts=[second_set], dbname=second_dbname)
    # check nodes after insertion
    first_nodes = await faiss_controller_patch.list_fact_nodes(first_dbname)
    second_nodes = await faiss_controller_patch.list_fact_nodes(second_dbname)
    assert first_nodes
    assert second_nodes
    # single node deletion
    first_fact_id = first_nodes[0].node_id
    await faiss_controller_patch.delete_fact_nodes(dbname=first_dbname, fact_ids=[first_fact_id])
    first_nodes = await faiss_controller_patch.list_fact_nodes(first_dbname)
    assert len(first_nodes) == 1
    # delete all nodes
    await faiss_controller_patch.delete_fact_nodes(dbname=second_dbname)
    second_nodes = await faiss_controller_patch.list_fact_nodes(second_dbname)
    assert not (second_nodes)


@pytest.mark.integration
@pytest.mark.anyio
async def test_chroma_vectorstore_passes_tables_sync_and_retrieve(
    chroma_controller_patch: "BaseController",
):
    """
    Integration test: sync and retrieve table nodes on chroma vectorstore.
    Requires: ChromaDB server, embedding server, HeavyDB
    """
    from heavyiq.langchain.heavydb import HeavyDB

    db_name = "heavyai"
    heavydb = await HeavyDB.from_env_async(db_name=db_name)

    await chroma_controller_patch.sync_table_nodes(heavydb, force=True)

    nodes = await chroma_controller_patch.list_table_nodes(db_name)

    assert len(nodes) > 0
    assert all(
        n.metadata["type"] == "table" and n.metadata["dbname"] == db_name
        for n in nodes
    )


@pytest.mark.integration
@pytest.mark.anyio
async def test_faiss_vectorstore_passes_tables_sync_and_retrieve(
    faiss_controller_patch: "BaseController",
):
    """
    Integration test: sync and retrieve table nodes on faiss vectorstore.
    Requires: embedding server, HeavyDB
    """
    from heavyiq.langchain.heavydb import HeavyDB

    db_name = "heavyai"
    heavydb = await HeavyDB.from_env_async(db_name=db_name)

    await faiss_controller_patch.sync_table_nodes(heavydb, force=True)

    nodes = await faiss_controller_patch.list_table_nodes(db_name)

    assert len(nodes) > 0
    assert all(
        n.metadata["type"] == "table" and n.metadata["dbname"] == db_name
        for n in nodes
    )
