import json
from enum import Enum
from typing import Any
from dataclasses import dataclass, asdict
from fastapi_socketio import SocketManager
from langchain.callbacks.base import AsyncCallbackHandler

from langchain.schema.agent import AgentAction, AgentFinish
from langchain.schema.output import LLMResult


class SocketIOEvent(Enum):
    """
    Socket IO Events which will be emitted upon executing
    relevant handler functions inside a CallBack class.
    """

    LLM_START = "llm_start"
    LLM_NEW_TOKEN = "llm_new_token"
    LLM_ERROR = "llm_error"
    LLM_END = "llm_end"
    CHAIN_START = "chain_start"
    CHAIN_ERROR = "cahin_error"
    CHAIN_END = "chain_end"
    TOOL_START = "tool_start"
    TOOL_ERROR = "tool_error"
    TOOL_END = "tool_end"
    AGENT_ACTION = "agent_action"
    AGENT_FINISH = "agent_finish"


class EventStatus(Enum):
    """
    Event status enum.
    """

    success = "success"
    error = "error"
    info = "info"


@dataclass
class SocketEventInfo:
    """
    Class comprises the data that is going to be sent to the SocketIO client.
    """

    # name of the event
    event: str
    # text to be displayed
    text: str
    # additional data
    data: dict[str, Any]
    # status to identitify whether it's a success or error or info event
    status: str


class AsyncSocketCallbackHandler(AsyncCallbackHandler):
    """
    This class is responsible for emitting events related to agents, chains, and llms to the connected SocketIO clients.
    """

    def __init__(self, socket_manager: SocketManager, sid: str) -> None:
        """
        Init func.

        Args:
            socket_manager (SocketManager): socket manager responsible for emitting events.
            sid (str): socket_id/room_id/client_id to which the data is going to be emitted.
        """
        self._socket_manager = socket_manager
        self._sid = sid

    async def send(self, event_info: SocketEventInfo):
        """
        Emit event to a particular sid.
        """
        await self._socket_manager.emit(event_info.event, asdict(event_info), to=self._sid)

    async def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> Any:
        """Run when LLM starts running."""
        event = SocketEventInfo(
            event=SocketIOEvent.LLM_START.value, text="LLM Started!", data=serialized, status=EventStatus.info.value
        )
        await self.send(event)

    async def on_llm_error(self, error: Exception | KeyboardInterrupt, **kwargs: Any) -> Any:
        """Run when LLM errors."""
        event = SocketEventInfo(
            event=SocketIOEvent.LLM_ERROR.value, status=EventStatus.error.value, text=f"LLM Error! {error}", data={}
        )
        await self.send(event)

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        event = SocketEventInfo(
            event=SocketIOEvent.LLM_END.value,
            status=EventStatus.info.value,
            text=f"LLM Finished! Result: {response.llm_output}",
            data={},
        )
        await self.send(event)

    async def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any) -> None:
        """Print out that we are entering a chain."""
        class_name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
        event = SocketEventInfo(
            event=SocketIOEvent.CHAIN_START.value,
            status=EventStatus.info.value,
            text=f"Entering new {class_name} chain...",
            data=serialized,
        )
        await self.send(event)

    async def on_chain_error(self, error: Exception | KeyboardInterrupt, **kwargs: Any) -> Any:
        """Run when chain errors."""
        event = SocketEventInfo(
            event=SocketIOEvent.CHAIN_ERROR.value, status=EventStatus.error.value, text=f"Chain Error! {error}", data={}
        )
        await self.send(event)

    async def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""
        event = SocketEventInfo(
            event=SocketIOEvent.CHAIN_END.value, status=EventStatus.info.value, text="Chain Finished!", data=outputs
        )
        await self.send(event)

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Run when tool starts running."""
        event = SocketEventInfo(
            event=SocketIOEvent.TOOL_START.value,
            status=EventStatus.info.value,
            text=f"Tool \"{serialized['name']}\" Started! Input: {input_str}",
            data=serialized,
        )
        await self.send(event)

    async def on_tool_error(
        self,
        error: Exception | KeyboardInterrupt,
        **kwargs: Any,
    ) -> None:
        """Run when tool errors."""
        event = SocketEventInfo(
            event=SocketIOEvent.TOOL_ERROR.value, status=EventStatus.error.value, text=f"Tool Error! {error}", data={}
        )
        await self.send(event)

    async def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """If not the final action, print out observation."""
        status = EventStatus.error.value if output.startswith("Exception") else EventStatus.success.value
        event = SocketEventInfo(
            status=status,
            event=SocketIOEvent.TOOL_END.value,
            text=f"Tool Ends! Output: {output}",
            data={},
        )
        await self.send(event)

    async def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        action_log = json.loads(action.log)
        event = SocketEventInfo(
            event=SocketIOEvent.AGENT_ACTION.value,
            status=EventStatus.info.value,
            text=f'Thought: {action_log.get("thought", "")}',
            data=asdict(action),
        )
        await self.send(event)

    async def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """Run on agent end."""
        event = SocketEventInfo(
            event=SocketIOEvent.AGENT_FINISH.value,
            status=EventStatus.success.value,
            text=f"Agent Finished! {finish.return_values}",
            data={"log": finish.log},
        )
        await self.send(event)
