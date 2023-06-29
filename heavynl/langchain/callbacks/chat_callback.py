import time

from typing import Any
from flask_socketio import emit
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)


class StreamingChatCallbackHandler(FinalStreamingStdOutCallbackHandler):
    buffer: list[tuple[str, float]] = []
    stop_token = "#!stop!#"

    def __init__(
        self,
        *,
        answer_prefix_tokens: list[str] | None = None,
        answer_stop_tokens: list[str] | None = None,
        strip_tokens: bool = True,
        stream_prefix: bool = False,
    ) -> None:
        answer_prefix_tokens = answer_prefix_tokens or [
            "action",
            '":',
            ' "',
            "Final",
            " Answer",
            '",\n',
            "   ",
            ' "',
            "action",
            "_input",
            '":',
            ' "',
        ]
        super().__init__(
            answer_prefix_tokens=answer_prefix_tokens, strip_tokens=strip_tokens, stream_prefix=stream_prefix
        )
        self.answer_stop_reached = False
        self.answer_stop_tokens = answer_stop_tokens or ['"', "\n"]

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        super().on_llm_start(serialized, prompts, **kwargs)
        self.buffer = []
        emit("startToken", {"data": ""})

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        # Remember the last n tokens, where n = len(answer_prefix_tokens)
        self.last_tokens.append(token)
        if len(self.last_tokens) > len(self.answer_prefix_tokens):
            self.last_tokens.pop(0)

        # Check if the last n tokens match the answer_prefix_tokens list ...
        if self.last_tokens == self.answer_prefix_tokens:
            self.answer_reached = True
            # Do not print the last token in answer_prefix_tokens,
            # as it's not part of the answer yet
            return

        # join last two token and then match against
        stop_string = "".join(self.answer_stop_tokens)
        if self.last_tokens[-1].endswith(stop_string):
            self.answer_stop_reached = True
            token = self.last_tokens[-1].rstrip(stop_string)
            self.add_to_buffer(token)

        # ... if yes, then append tokens to buffer
        if self.answer_reached and not self.answer_stop_reached:
            self.add_to_buffer(token)

    def add_to_buffer(self, token: str) -> None:
        now = time.time()
        self.buffer.append((token, now))

        if token != self.stop_token:
            emit("newToken", {"data": token})

    def stream_chars(self):
        while True:
            # when we didn't receive any token yet, just continue
            if len(self.buffer) == 0:
                continue

            token, timestamp = self.buffer.pop(0)

            if token != self.stop_token:
                for character in token:
                    yield (character, timestamp)
            else:
                break
