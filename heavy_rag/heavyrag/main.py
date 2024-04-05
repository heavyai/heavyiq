import os

from heavyrag.database.etl import ask_db_index
from heavyrag.documents.etl import ask_doc_index

# disable anonymized telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "false"
