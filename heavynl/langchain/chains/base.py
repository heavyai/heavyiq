from langchain.chains.base import Chain
from langchain.callbacks.manager import CallbackManagerForChainRun
from heavynl.logging_utils import get_heavynl_logger


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


class FileCallbackHandlerForChainMixin:
    """
    Mixin class for chains which helps to add FileHandler callback if no callbacks are specified.
    """

    def __init__(self, *args, **kwargs):
        logger = get_heavynl_logger()
        # add langchain_cb_handler handler if callbacks are not specified
        kwargs["callbacks"] = kwargs.get("callbacks", None) or [logger.langchain_cb_handler(to_stdout=False)]
        super().__init__(*args, **kwargs)  # type: ignore
