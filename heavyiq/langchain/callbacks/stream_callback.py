from typing import Any
from uuid import UUID
from langchain.schema.output import LLMResult
from langchain.callbacks import AsyncIteratorCallbackHandler


class StreamingChainCallbackHandler(AsyncIteratorCallbackHandler):
    """
    Chain callback handler mainly used to stream llm tokens or text as response through iter method.
    """

    async def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any
    ) -> None:
        self.done.clear()

    async def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        pass

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        pass

    async def on_chain_error(
        self,
        error: Exception | KeyboardInterrupt,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any
    ) -> None:
        self.done.set()

    async def on_text(
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        event_type: str = "step",
        **kwargs: Any
    ) -> None:
        self.queue.put_nowait({"type": event_type, "text": text})  # type: ignore

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if token is not None and token != "":
            self.queue.put_nowait({"type": "next_token", "text": token})  # type: ignore

    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any
    ) -> None:
        self.done.set()
