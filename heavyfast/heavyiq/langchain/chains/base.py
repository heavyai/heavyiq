from langchain.chains.base import Chain
from langchain.callbacks.manager import CallbackManagerForChainRun


class BaseChain(Chain):
    """
    Base class for the custom chain classes.
    """

    def write_callback_message(
        self, message: str, run_manager: CallbackManagerForChainRun | None = None, color: str | None = None
    ):
        """
        Writes back the messages to terminal console.
        """
        if run_manager:
            run_manager.on_text(message, color=color, verbose=self.verbose)
