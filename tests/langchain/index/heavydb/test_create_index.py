import pytest
from typing import Callable, Any
from unittest.mock import patch
from tests.langchain.index import ConsistentFakeEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.schema import Document
from tests import aoverride_config
from heavyiq.config import HeavyIQConfig
from heavyiq.langchain.index.heavydb.create_index import acreate_index_if_nonexistent


async def mock_agenerate_table_documents() -> list[str]:
    return []


async def mock_agenerate_table_documents_with_table() -> list[str]:
    return ["movie_actress"]


def mock_get_vectorstore_index_creator(metadata_index_dir: str) -> VectorstoreIndexCreator:
    return VectorstoreIndexCreator(
        vectorstore_kwargs={"persist_directory": metadata_index_dir},
        embedding=ConsistentFakeEmbeddings(),
    )


async def mock_aread_table_documents(doc: Document):
    yield doc


@pytest.mark.asyncio
@aoverride_config
@patch(
    "heavyiq.langchain.index.heavydb.create_index.agenerate_table_documents", side_effect=mock_agenerate_table_documents
)
@patch(
    "heavyiq.langchain.index.heavydb.create_index.get_vectorstore_index_creator",
    side_effect=mock_get_vectorstore_index_creator,
)
async def test_should_create_heavydb_index_if_metadata_index_dir_does_not_exsits(
    mock_a: Callable,
    mock_b: Callable,
    heavyiq_config: HeavyIQConfig,
    sample_table_document: Document,
    testdb_delete_on_teardown: Any,
):
    with patch("heavyiq.langchain.index.heavydb.create_index.aread_table_documents") as mock_async_generator:
        mock_async_generator.return_value = mock_aread_table_documents(sample_table_document)
        heavydb_index = await acreate_index_if_nonexistent()
        assert heavydb_index


@pytest.mark.asyncio
@aoverride_config
@patch(
    "heavyiq.langchain.index.heavydb.create_index.agenerate_table_documents",
    side_effect=mock_agenerate_table_documents_with_table,
)
@patch(
    "heavyiq.langchain.index.heavydb.create_index.get_vectorstore_index_creator",
    side_effect=mock_get_vectorstore_index_creator,
)
async def test_should_create_heavydb_index_if_metadata_index_dir_exsits(
    mock_a: Callable, mock_b: Callable, heavyiq_config: HeavyIQConfig, sample_table_document: Document, testdb: Any
):
    with patch("heavyiq.langchain.index.heavydb.create_index.aread_table_documents") as mock_async_generator:
        mock_async_generator.return_value = mock_aread_table_documents(sample_table_document)
        heavydb_index = await acreate_index_if_nonexistent()
        assert heavydb_index
