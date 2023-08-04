from typing import Any
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction


# First, define custom callback handler implementations
class ConvoAgentCallbackHandler(BaseCallbackHandler):
    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        pass
