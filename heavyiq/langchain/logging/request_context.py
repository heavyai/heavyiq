from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from typing import Any, Optional

from langchain.agents import AgentExecutor
from langchain.chains.base import Chain
from langchain_core.runnables import Runnable, RunnableConfig, RunnableSerializable
from langchain_community.callbacks import OpenAICallbackHandler, get_openai_callback
from langchain_core.tracers.context import collect_runs

from heavyiq import VERSION_TAG
from heavyiq.config import get_config

# from .database import Session, RequestLog
from .enums import LangChainType


class RequestContext:
    def __init__(self, langchain_type: LangChainType, langchain_name: str, model: str, input: Optional[dict]):
        self.langchain_type = langchain_type
        self.langchain_name = langchain_name
        self.model = model
        self.input = input
        self.start_time = datetime.now()
        self.output: dict[Any, Any] | None = None
        self.prompt_tokens: int | None = None
        self.completion_tokens: int | None = None
        self.total_tokens: int | None = None
        self.successful_requests: int | None = None
        self.total_cost: float | None = None
        self.output = None
        self.agent_log = None
        self.error_msg: str | None = None

    async def __aenter__(self) -> "RequestContext":
        return self

    def __enter__(self) -> "RequestContext":
        return self

    async def __aexit__(self, *args) -> Any:
        pass

    def __exit__(self, *args: Any) -> None:
        pass

    def add_input(self, input: dict) -> None:
        self.input = input

    def success(
        self, output: dict[Any, Any], cb: Optional[OpenAICallbackHandler] = None, agent_log: Optional[str] = None
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
        config = get_config()
        if config.log_to_stdout:
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
            self.input,  # type: ignore
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
    pass
    # new_request = RequestLog(
    #    langchain_type=langchain_type,
    #    langchain_name=langchain_name,
    #    start_time=start_time,
    #    end_time=end_time,
    #    input=input,
    #    output=output,
    #    model=model,
    #    error=error,
    #    agent_log=agent_log,
    #    prompt_tokens=prompt_tokens,
    #    completion_tokens=completion_tokens,
    #    total_tokens=total_tokens,
    #    successful_requests=successful_requests,
    #    total_cost=total_cost,
    # )

    # with Session() as session:
    #     session.add(new_request)
    #     session.commit()


@contextmanager
def log_request_ctx(
    langchain_type: LangChainType, langchain_name: str, model: str = "", input: Optional[dict] = None
) -> Generator[RequestContext, None, None]:
    ctx = RequestContext(langchain_type, langchain_name, model, input)
    yield ctx


@contextmanager
def chain_log_request_ctx(
    langchain_name: str, model: str = "", input: Optional[dict] = None
) -> Generator[RequestContext, None, None]:
    ctx = RequestContext(LangChainType.Chain, langchain_name, model, input)
    yield ctx


@asynccontextmanager
async def chain_log_request_ctx_async(
    langchain_name: str, model: str = "", input: Optional[dict] = None
) -> AsyncGenerator[RequestContext, None]:
    ctx = RequestContext(LangChainType.Chain, langchain_name, model, input)
    yield ctx


@asynccontextmanager
async def runnable_log_request_ctx(
    runnable_name: str, model: str = "", input: Optional[dict] = None
) -> AsyncGenerator[RequestContext, None]:
    ctx = RequestContext(LangChainType.RunnableChain, runnable_name, model, input)
    yield ctx


@contextmanager
def agent_log_request_ctx(
    langchain_name: str, model: str = "", input: Optional[dict] = None
) -> Generator[RequestContext, None, None]:
    ctx = RequestContext(LangChainType.Agent, langchain_name, model, input)
    yield ctx


def log_chain_call(chain: Chain, input: str | dict, model: str = "", chain_name: Optional[str] = None) -> dict:
    from heavyiq.langchain.utils import is_langsmith_active

    if isinstance(input, str):
        input = {chain.input_keys[0]: input}
    with chain_log_request_ctx(chain_name or chain._chain_type, model, input) as log_ctx:
        with get_openai_callback() as cb:
            try:
                output = chain(input, include_run_info=is_langsmith_active, tags=[VERSION_TAG])
                log_ctx.success(output, cb)
                return output
            except Exception as e:
                log_ctx.error(str(e), cb)
                raise e


async def log_chain_call_async(
    chain: Chain, input: str | dict, model: str = "", chain_name: Optional[str] = None
) -> dict:
    from heavyiq.langchain.utils import is_langsmith_active

    if isinstance(input, str):
        input = {chain.input_keys[0]: input}
    async with chain_log_request_ctx_async(chain_name or chain._chain_type, model, input) as log_ctx:
        with get_openai_callback() as cb:
            try:
                output = await chain.acall(input, include_run_info=is_langsmith_active, tags=[VERSION_TAG])
                log_ctx.success(output, cb)
                return output
            except Exception as e:
                log_ctx.error(str(e), cb)
                raise e


async def log_chain_runnable(
    runnable: RunnableSerializable | Runnable,
    input: dict | str,
    config: RunnableConfig | None = None,
    model: str = "",
) -> str:
    """
    Helps to invoke and log tokens used by the runnable.
    """
    config = config or {}
    async with runnable_log_request_ctx("", model, input) as log_ctx:  # type: ignore
        with get_openai_callback() as cb:
            with collect_runs() as clrs:  # context manager which helps to grab the runs list
                try:
                    tags = config.get("tags", [])
                    tags.append(VERSION_TAG)
                    config["tags"] = tags
                    output = await runnable.ainvoke(input, config=config)
                    # helps to grab the last run
                    last_run = clrs.traced_runs[0]
                    log_ctx.langchain_name = last_run.name
                    log_ctx.success(output, cb)
                    return output
                except Exception as e:
                    log_ctx.error(str(e), cb)
                    raise e


def log_agent_call(agent: AgentExecutor, input: str, model: str = "") -> dict:
    with agent_log_request_ctx("agent", model, {"input": input}) as log_ctx:
        with get_openai_callback() as cb:
            try:
                agent.return_intermediate_steps = True
                res = agent({"input": input})
                output = res["output"]
                intermediate_steps = res["intermediate_steps"]
                log_ctx.success(output, cb, "\n".join([str(step) for step in intermediate_steps]))
                return output
            except Exception as e:
                log_ctx.error(str(e), cb)
                raise e
