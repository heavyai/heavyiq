from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, Any

from langchain.agents.agent import AgentExecutor
from langchain.callbacks import OpenAICallbackHandler, get_openai_callback
from langchain.chains.base import Chain

from .database import request_log_table, engine
from .enums import LangChainType


class RequestContext:
    def __init__(self, langchain_type: LangChainType, langchain_name: str, model: str, input: Optional[dict]):
        self.langchain_type = langchain_type
        self.langchain_name = langchain_name
        self.model = model
        self.input = input
        self.start_time = datetime.now()
        self.output = None
        self.prompt_tokens = None
        self.completion_tokens = None
        self.total_tokens = None
        self.successful_requests = None
        self.total_cost = None
        self.output = None
        self.agent_log = None
        self.error_msg = None

    def __enter__(self) -> "RequestContext":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def add_input(self, input: dict) -> None:
        self.input = input

    def success(
        self, output: dict, cb: Optional[OpenAICallbackHandler] = None, agent_log: Optional[str] = None
    ) -> None:
        self.output = output
        if cb:
            self.add_openai_cb(cb)
        self.commit()

    def add_openai_cb(self, cb: OpenAICallbackHandler) -> None:
        self.prompt_tokens = cb.prompt_tokens
        self.completion_tokens = cb.completion_tokens
        self.total_tokens = cb.total_tokens
        self.successful_requests = cb.successful_requests
        self.total_cost = cb.total_cost
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Successful Requests: {cb.successful_requests}")
        print(f"Total Cost (USD): ${cb.total_cost}")

    def error(self, error: str, cb: Optional[OpenAICallbackHandler] = None, agent_log: Optional[str] = None) -> None:
        self.error_msg = error
        if cb:
            self.add_openai_cb(cb)
        self.commit()

    def commit(
        self,
    ) -> None:
        end_time = datetime.now()
        log_request(
            self.langchain_type,
            self.langchain_name,
            self.start_time,
            end_time,
            self.input,
            self.model,
            self.output,
            self.error_msg,
            self.agent_log,
            self.prompt_tokens,
            self.completion_tokens,
            self.total_tokens,
            self.successful_requests,
            self.total_cost,
        )


def log_request(
    langchain_type: LangChainType,
    langchain_name: str,
    start_time: datetime,
    end_time: datetime,
    input: dict,
    model: str,
    output: Optional[dict] = None,
    error: Optional[str] = None,
    agent_log: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    successful_requests: Optional[int] = None,
    total_cost: Optional[float] = None,
) -> None:
    with engine.begin() as connection:
        row = request_log_table.insert().values(
            langchain_type=langchain_type,
            langchain_name=langchain_name,
            start_time=start_time,
            end_time=end_time,
            input=input,
            output=output,
            model=model,
            error=error,
            agent_log=agent_log,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            successful_requests=successful_requests,
            total_cost=total_cost,
        )
        connection.execute(row)


@contextmanager
def log_request_ctx(
    langchain_type: LangChainType, langchain_name: str, model: str, input: Optional[dict] = None
) -> Generator[RequestContext, None, None]:
    ctx = RequestContext(langchain_type, langchain_name, model, input)
    yield ctx


@contextmanager
def chain_log_request_ctx(
    langchain_name: str, model: str, input: Optional[dict] = None
) -> Generator[RequestContext, None, None]:
    ctx = RequestContext(LangChainType.Chain, langchain_name, model, input)
    yield ctx


@contextmanager
def agent_log_request_ctx(
    langchain_name: str, model: str, input: Optional[dict] = None
) -> Generator[RequestContext, None, None]:
    ctx = RequestContext(LangChainType.Agent, langchain_name, model, input)
    yield ctx


def log_chain_call(chain: Chain, input: str | dict, model: str, chain_name: Optional[str]) -> dict:
    if isinstance(input, str):
        input = {chain.input_keys[0]: input}
    with chain_log_request_ctx(chain_name or chain._chain_type, model, input) as log_ctx:
        with get_openai_callback() as cb:
            try:
                output = chain(input)
                log_ctx.success(output, cb)
                return output
            except Exception as e:
                log_ctx.error(str(e), cb)
                raise e


def log_agent_call(agent: AgentExecutor, input: str, model: str) -> dict:
    with agent_log_request_ctx("agent", model, {"input": input}) as log_ctx:
        with get_openai_callback() as cb:
            try:
                output = agent(input)
                log_ctx.success(output, cb)
                return output
            except Exception as e:
                log_ctx.error(str(e), cb)
                raise e
