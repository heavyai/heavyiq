from langchain.chains.base import Chain


from langchain_core.callbacks import CallbackManagerForChainRun

from langchain_core.callbacks import AsyncCallbackManagerForChainRun
from heavyiq.logging_utils import get_heavyiq_logger
from heavyiq.config import get_config


class FileCallbackHandlerForChainMixin:
    """
    Mixin class for chains which helps to add FileHandler callback if no callbacks are specified.
    """

    def __init__(self, *args, **kwargs):
        logger = get_heavyiq_logger()
        config = get_config()
        callbacks = kwargs.get("callbacks", None)
        # add langchain_cb_handler handler if callbacks are not specified
        if callbacks is None:
            kwargs["callbacks"] = [logger.langchain_cb_handler(to_stdout=config.log_to_stdout)]
        super().__init__(*args, **kwargs)  # type: ignore


class BaseChain(FileCallbackHandlerForChainMixin, Chain):
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

    async def write_callback_message_async(
        self, message: str, run_manager: AsyncCallbackManagerForChainRun | None = None, color: str | None = None
    ) -> None:
        if run_manager:
            await run_manager.on_text(message, color=color, verbose=self.verbose)
