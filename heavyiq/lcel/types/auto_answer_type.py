from pydantic import Field

from .answer_type import AnswerChainOutputType
from .auto_sql_type import AutoSQLChainInputType


class AutoAnswerChainInputType(AutoSQLChainInputType):
    ...


class AutoAnswerChainOutputType(AnswerChainOutputType):
    tables: list[str] = Field(..., description="List of table names used.")
