from langchain.pydantic_v1 import Field

from .answer_type import NLtoAnswerChainOutputType
from .auto_sql_type import AutoSQLChainInputType


class AutoAnswerChainInputType(AutoSQLChainInputType):
    ...


class AutoAnswerChainOutputType(NLtoAnswerChainOutputType):
    tables: list[str] = Field(..., description="List of table names used.")
