# Test RAG router endpoints (e2e tests - require full app stack)

import uuid

import pytest
from fastapi.testclient import TestClient

from heavyiq.config import HeavyIQConfig


# Mark all tests in this module as e2e (end-to-end API tests)
pytestmark = pytest.mark.e2e


def _unique_snippet(text: str) -> str:
    """Generate a unique snippet by appending a UUID."""
    return f"{text} [test-{uuid.uuid4().hex[:8]}]"


class TestSnippetEndpoints:
    """Test snippets/facts CRUD endpoints."""

    @pytest.mark.anyio
    async def test_insert_snippet(self, heavyiq_config: HeavyIQConfig, client: TestClient, session_id: str):
        """Test inserting a new snippet."""
        payload = {
            "session_id": session_id,
            "snippet": _unique_snippet("The heavyai_us_states table contains information about US states including population and area."),
        }
        response = client.post("/rag/snippets/insert", json=payload)
        assert response.status_code == 200
        response_json = response.json()
        assert "snippet_id" in response_json
        assert response_json["snippet_id"] is not None

        # Store for cleanup
        self._snippet_id = response_json["snippet_id"]

    @pytest.mark.anyio
    async def test_list_snippets(self, heavyiq_config: HeavyIQConfig, client: TestClient, session_id: str):
        """Test listing all snippets."""
        payload = {
            "session_id": session_id,
            "verify": False,
        }
        response = client.post("/rag/snippets/list", json=payload)
        assert response.status_code == 200
        response_json = response.json()
        assert "snippets" in response_json
        assert isinstance(response_json["snippets"], list)

    @pytest.mark.anyio
    async def test_bulk_insert_snippets(self, heavyiq_config: HeavyIQConfig, client: TestClient, session_id: str):
        """Test bulk inserting snippets."""
        payload = {
            "session_id": session_id,
            "snippets": [
                _unique_snippet("Snippet 1: The database contains geographic data."),
                _unique_snippet("Snippet 2: Population data is available for all states."),
            ],
        }
        response = client.post("/rag/snippets/bulk-insert", json=payload)
        assert response.status_code == 200
        response_json = response.json()
        assert "snippet_ids" in response_json
        assert len(response_json["snippet_ids"]) == 2

    @pytest.mark.anyio
    async def test_search_snippets(self, heavyiq_config: HeavyIQConfig, client: TestClient, session_id: str):
        """Test searching for snippets."""
        payload = {
            "session_id": session_id,
            "question": "What tables contain state information?",
            "top_k": 5,
        }
        response = client.post("/rag/snippets/search", json=payload)
        assert response.status_code == 200
        response_json = response.json()
        assert "nodes" in response_json
        assert isinstance(response_json["nodes"], list)

    @pytest.mark.anyio
    async def test_get_snippet_index_nodes(self, heavyiq_config: HeavyIQConfig, client: TestClient, session_id: str):
        """Test getting snippet nodes from index."""
        payload = {
            "session_id": session_id,
        }
        response = client.post("/rag/snippets/index/nodes", json=payload)
        assert response.status_code == 200
        response_json = response.json()
        assert "nodes" in response_json

    @pytest.mark.anyio
    async def test_sync_snippets(self, heavyiq_config: HeavyIQConfig, client: TestClient, session_id: str):
        """Test syncing snippet nodes."""
        payload = {
            "session_id": session_id,
            "force": False,
        }
        response = client.post("/rag/snippets/sync", json=payload)
        assert response.status_code == 200
        response_json = response.json()
        assert response_json["success"] is True

    @pytest.mark.anyio
    async def test_ask_snippets(self, heavyiq_config: HeavyIQConfig, client: TestClient, session_id: str):
        """Test asking questions about snippets."""
        # First insert a snippet
        insert_payload = {
            "session_id": session_id,
            "snippet": _unique_snippet("The heavyai_us_states table has columns: state_name, population, area_sq_miles."),
        }
        insert_response = client.post("/rag/snippets/insert", json=insert_payload)
        assert insert_response.status_code == 200

        # Now ask about it
        ask_payload = {
            "session_id": session_id,
            "question": "What columns does the states table have?",
            "only_retrieve": False,
            "do_evaluate": False,
            "use_reranker": False,
        }
        response = client.post("/rag/snippets/ask", json=ask_payload)
        assert response.status_code == 200
        response_json = response.json()
        # Either has answer or sources
        assert "answer" in response_json or "sources" in response_json


class TestTableEndpoints:
    """Test table-related RAG endpoints."""

    @pytest.mark.anyio
    async def test_derive_table_names(self, heavyiq_config: HeavyIQConfig, client: TestClient, session_id: str):
        """Test deriving relevant table names from question."""
        payload = {
            "session_id": session_id,
            "question": "How many states are there in the US?",
        }
        response = client.post("/rag/tables/names", json=payload)
        assert response.status_code == 200
        response_json = response.json()
        assert isinstance(response_json, list)

    @pytest.mark.anyio
    async def test_list_table_nodes(self, heavyiq_config: HeavyIQConfig, client: TestClient, session_id: str):
        """Test listing table nodes from index."""
        payload = {
            "session_id": session_id,
        }
        response = client.post("/rag/tables/index/nodes", json=payload)
        assert response.status_code == 200
        response_json = response.json()
        assert "nodes" in response_json

    @pytest.mark.anyio
    async def test_sync_table_nodes(self, heavyiq_config: HeavyIQConfig, client: TestClient, session_id: str):
        """Test syncing table nodes."""
        payload = {
            "session_id": session_id,
            "force": False,
        }
        response = client.post("/rag/tables/sync", json=payload)
        assert response.status_code == 200
        response_json = response.json()
        assert response_json["success"] is True

    @pytest.mark.anyio
    async def test_search_table_nodes(self, heavyiq_config: HeavyIQConfig, client: TestClient, session_id: str):
        """Test searching table nodes."""
        payload = {
            "session_id": session_id,
            "question": "tables with state information",
            "top_k": 5,
        }
        response = client.post("/rag/tables/search", json=payload)
        assert response.status_code == 200
        response_json = response.json()
        assert "nodes" in response_json
        assert isinstance(response_json["nodes"], list)


class TestDocumentEndpoints:
    """Test document-related RAG endpoints."""

    @pytest.mark.anyio
    async def test_list_documents(self, heavyiq_config: HeavyIQConfig, client: TestClient, session_id: str):
        """Test listing uploaded documents."""
        payload = {
            "session_id": session_id,
        }
        response = client.post("/rag/documents/list", json=payload)
        assert response.status_code == 200
        response_json = response.json()
        assert "documents" in response_json
        assert isinstance(response_json["documents"], list)

