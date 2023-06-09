class HeavyNLBaseException(Exception):
    """
    Base exception class for all the heavynl exceptions.
    """

    def __init__(self, message: str | None = None):
        self.message = message

    def get_child_class(self):
        return type(self)

    def __str__(self):
        return f"{self.get_child_class()}: {self.message}"


class NLtoSQLException(HeavyNLBaseException):
    pass
