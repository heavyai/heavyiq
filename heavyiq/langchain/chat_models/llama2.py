from langchain.chat_models import ChatOpenAI


class Llama2Chat(ChatOpenAI):
    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "llama2-chat"
