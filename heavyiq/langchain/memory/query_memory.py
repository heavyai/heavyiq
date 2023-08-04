from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema import BaseMessage, get_buffer_string
from typing import Any


class HeavyIQQueryBufferWindowMemory(BaseChatMemory):
    """Memory class for storing information about sql queries."""

    # Define key to pass information about queries into prompt.
    input_key: str = "question"
    output_key: str = "query"
    memory_key: str = "queries"
    human_prefix: str = "Recent Question"
    ai_prefix: str = "Recently executed SQL"
    k: int = 5

    @property
    def buffer(self) -> list[BaseMessage]:
        """String buffer of memory."""
        return self.chat_memory.messages

    @property
    def memory_variables(self) -> list[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str], manual_save: bool = False) -> None:
        """
        Save context from this conversation to buffer.
        As for now, this should saves the messages upon manual save.
        """
        if manual_save:
            input_str, output_str = self._get_input_output(inputs, outputs)
            self.chat_memory.add_user_message(input_str)
            self.chat_memory.add_ai_message(output_str)

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, str]:
        """Return history buffer."""

        buffer: Any = self.buffer[-self.k * 2 :] if self.k > 0 else []
        if not self.return_messages:
            buffer = get_buffer_string(
                buffer,
                human_prefix=self.human_prefix,
                ai_prefix=self.ai_prefix,
            )
        return {self.memory_key: buffer}
