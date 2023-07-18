from langchain.chains.base import Chain
from langchain.callbacks.manager import CallbackManagerForChainRun, AsyncCallbackManagerForChainRun


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

    async def write_callback_message_async(
        self, message: str, run_manager: AsyncCallbackManagerForChainRun | None = None, color: str | None = None
    ) -> None:
        if run_manager:
            await run_manager.on_text(message, color=color, verbose=self.verbose)
