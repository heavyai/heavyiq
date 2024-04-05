from enum import Enum


class DocType(Enum):
    TXT = "TXT"
    PDF = "PDF"


class DocMetadataKeys(Enum):
    """
    Metadata keys for llama-index document.
    """

    DOC_TYPE = "doc_type"
    FILE_NAME = "file_name"
    DATABASE_DOC_ID = "database_doc_id"
    TYPE = "type"
