from typing import Any

from confz import DataSource

from heavyiq.api.models import GenerateTableMetadataRequest, QueryRequest, QuestionRequest
from unittest.mock import patch
from heavyiq.langchain import HeavyDB
from heavyiq.config.config_schema import HeavyIQConfig
from fastapi import Depends


def override_heavydb_client():
    print("Creating heavydb mock client")
    new_source = DataSource(
        data={
            "heavydb_dbname": "heavynl",
            "heavydb_username": "admin",
            "heavydb_password": "HyperInteractive",
            "heavydb_host": "10.2.1.33",
            "heavydb_port": "6274",
            "heavydb_protocol": "binary",
            "openai_api_key": "",
        }
    )
    try:
        # Use patch to mock the get_config function
        with HeavyIQConfig.change_config_sources(new_source):
            with patch("heavyiq.langchain.heavydb.get_config", return_value=HeavyIQConfig()):
                yield HeavyDB.from_env()
    except Exception as e:
        print(f"An error occurred during mock heavydb creation: {e}")
        yield None

    print("Tearing heavydb mock client")


def override_heavydb_client_for_query_request(
    query_request: QueryRequest, db: HeavyDB | None = Depends(override_heavydb_client)
) -> tuple[QueryRequest, HeavyDB]:
    assert db
    return query_request, db


def override_heavydb_client_for_question_request(
    query_request: QuestionRequest, db: HeavyDB | None = Depends(override_heavydb_client)
) -> tuple[QuestionRequest, HeavyDB]:
    assert db
    return (query_request, db)


def override_valid_table_metadata_db_session(
    query_request: GenerateTableMetadataRequest, db: HeavyDB | None = Depends(override_heavydb_client)
) -> tuple[GenerateTableMetadataRequest, HeavyDB]:
    assert db
    return (query_request, db)
