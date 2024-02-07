class HeavyIQBaseException(Exception):
    """
    Base exception class for all the heavyiq exceptions.
    """

    def __init__(self, message: str | None = None):
        self.message = message

    def get_child_class(self) -> type:
        return type(self)

    def __str__(self) -> str:
        return f"{self.get_child_class()}: {self.message}"


class NLtoSQLException(HeavyIQBaseException):
    def __init__(self, message: str | None = None, failed_sql: str | None = None):
        super().__init__(message)
        self.failed_sql = failed_sql

    def __str__(self) -> str:
        base_message = super().__str__()
        if self.failed_sql is not None:
            return f"{base_message}\nFailed SQL: {self.failed_sql}"
        return base_message


class GenerateTableMetadataException(HeavyIQBaseException):
    pass


class NLtoAnswerException(HeavyIQBaseException):
    pass


class NLtoSQLPredefinedException(HeavyIQBaseException):
    pass
