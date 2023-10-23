from langchain.chains.base import Chain
from langchain.callbacks.manager import CallbackManagerForChainRun, AsyncCallbackManagerForChainRun, _ahandle_event
from heavyiq.logging_utils import get_heavyiq_logger
from heavyiq.config import get_config
from langchain.callbacks import AsyncIteratorCallbackHandler


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
            non_stream_handlers = [
                handler for handler in run_manager.handlers if not isinstance(handler, AsyncIteratorCallbackHandler)
            ]
            await _ahandle_event(
                non_stream_handlers,  # type: ignore
                "on_text",
                None,
                message,
                run_id=run_manager.run_id,
                parent_run_id=run_manager.parent_run_id,
                tags=run_manager.tags,
                verbose=self.verbose,
            )

    async def stream_callback_message(
        self, message: str, run_manager: AsyncCallbackManagerForChainRun | None = None, event_type: str = "step"
    ) -> None:
        """
        Put message for streaming.
        """
        if run_manager:
            stream_handlers = [
                handler for handler in run_manager.handlers if isinstance(handler, AsyncIteratorCallbackHandler)
            ]
            await _ahandle_event(
                stream_handlers,  # type: ignore
                "on_text",
                None,
                message,
                run_id=run_manager.run_id,
                parent_run_id=run_manager.parent_run_id,
                tags=run_manager.tags,
                event_type=event_type,
                verbose=self.verbose,
            )
