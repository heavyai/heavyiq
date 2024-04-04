import pytest

from heavyrag.database.read import HeavyDBReader


@pytest.fixture(scope="package", autouse=True)
def heavydb_reader():
    """
    Fixture that supposed to return HeavyDBReader object.
    """
    return HeavyDBReader(
        user="admin",
        password="HyperInteractive",
        host="10.2.1.33",
        port=6274,
        dbname="heavyiq",
    )
