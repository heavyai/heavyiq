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
    pass
