import pytest
from tests import aoverride_config
from unittest.mock import patch
from heavyiq.langchain.index.heavydb.generate_table_documents import (
    aget_table_summary_document,
    aget_table_column_description_document,
    agenerate_table_documents,
)


@pytest.mark.asyncio
@aoverride_config
@patch("heavyiq.langchain.index.heavydb.generate_table_documents.get_llm")
async def test_aget_table_summary_document_should_pass(mock_llm, mock_heavy_db, heavyiq_config, fake_chat_llm):
    content, table = "USA States table summary", "usa_states"
    fake_chat_llm.content = content
    mock_llm.return_value = fake_chat_llm
    doc = await aget_table_summary_document(mock_heavy_db, table)
    assert doc.page_content == content
    assert doc.metadata["source"] == table


@pytest.mark.asyncio
@aoverride_config
@patch("heavyiq.langchain.index.heavydb.generate_table_documents.get_llm")
async def test_aget_table_column_description_document_should_pass(
    mock_llm, mock_heavy_db, heavyiq_config, fake_chat_llm
):
    content, table = "USA States column summary", "usa_states"
    fake_chat_llm.content = content
    mock_llm.return_value = fake_chat_llm
    doc = await aget_table_column_description_document(mock_heavy_db, table)
    assert doc.page_content == f"Column descriptions for {table} table:\n{content}"
    assert doc.metadata["source"] == table


@pytest.mark.asyncio
@aoverride_config
@patch("heavyiq.langchain.index.heavydb.generate_table_documents.get_table_names_from_documents", return_value=[])
@patch("heavyiq.langchain.index.heavydb.generate_table_documents.acreate_and_write_table_document", return_value=True)
async def test_agenerate_table_documents_should_pass(mock_a, mock_b, heavyiq_config):
    assert "usa_states" in await agenerate_table_documents()
