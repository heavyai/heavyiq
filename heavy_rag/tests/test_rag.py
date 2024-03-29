from unittest.mock import patch
from heavyrag.main import ask


@patch("heavyrag.main.get_heavydb_reader")
def test_should_pass_rag_ask_function(mock, heavydb_reader):
    mock.return_value = heavydb_reader
    tables = ask("dummy_session", "How many states are there")
    assert "usa_states" == tables[0]
