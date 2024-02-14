from unittest.mock import patch

from confz import FileSource
from fastapi import Depends

from heavyiq.api.models import GenerateTableMetadataRequest, QueryRequest, QuestionRequest
from heavyiq.config.config_schema import AppConfig
from heavyiq.langchain import HeavyDB


def override_heavydb_client():
    """
    HeavyDB instance created from the parameters defined on the passed config file.
    """
    print("Creating heavydb mock client")
    from tests.conftest import CONFIG_FILE

    assert CONFIG_FILE

    new_source = FileSource(file=CONFIG_FILE)
    try:
        # Use patch to mock the get_config function
        with AppConfig.change_config_sources(new_source):
            with patch("heavyiq.langchain.heavydb.get_config", return_value=AppConfig().iq):  # type: ignore
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
