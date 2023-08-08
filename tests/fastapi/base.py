from fastapi.testclient import TestClient

from heavyiq.api import create_app
from heavyiq.api.dependencies import valid_query_db_session, valid_question_db_session

from .dependencies import override_valid_query_db_session

app = create_app()
app.dependency_overrides[valid_query_db_session] = override_valid_query_db_session
app.dependency_overrides[valid_question_db_session] = override_valid_query_db_session
client = TestClient(app)
