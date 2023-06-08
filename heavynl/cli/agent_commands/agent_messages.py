from dataclasses import dataclass


class TerminalColors:
    """
    Helps to add colors to the terminal messages.
    """

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    RESET = "\033[0m"


@dataclass
class _ConvoMessage:
    """
    Base message class for Conversational Agent Messages.
    """

    type: str
    prefix: str
    text_color: TerminalColors
    message: str = ""
    is_first_message: bool = False

    def __post_init__(self):
        self.first_message_prefix = "(type EXIT to break loop)"
        if self.is_first_message:
            self.prefix = self.prefix.replace(":", f" {self.first_message_prefix}:")

    def colorize(self) -> str:
        """
        Helps to colorize the messages upon printing on the perminal.
        """
        return f"{self.text_color}{self.prefix} {TerminalColors.RESET}{self.message}"

    def __str__(self) -> str:
        return f"{self.prefix} {self.message}"

    def __repr__(self) -> str:
        return f"{self.prefix} {self.message}"


@dataclass
class HumanConvoMessage(_ConvoMessage):
    """
    Mesages which are fed as input by human.

    Args:
        _ConvoMessage (object): Conversation message base class
    """

    type: str = "HUMAN"
    prefix: str = "You:"
    text_color: TerminalColors = TerminalColors.YELLOW


@dataclass
class AIConvoMessage(_ConvoMessage):
    """
    Mesages which are fed as input by AI.

    Args:
        _ConvoMessage (object): Conversation message base class
    """

    type: str = "AI"
    prefix: str = "Assistant:"
    text_color: TerminalColors = TerminalColors.GREEN
