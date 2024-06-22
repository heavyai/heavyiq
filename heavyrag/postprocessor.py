import re
from abc import abstractproperty
from enum import Enum, auto
from functools import partial
from typing import Any, Callable, List, Literal, Optional, Union

from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.llms.llm import LLM
from llama_index.core.postprocessor import LLMRerank, SimilarityPostprocessor
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.schema import BaseNode, MetadataMode, NodeWithScore, QueryBundle
from llama_index.core.service_context import ServiceContext

from heavyiq.config import get_config

from .llm import get_rerank_llm
from .prompts import DEFAULT_RAG_RERANK_CHOICE_SELECT_PROMPT, RAG_RERANK_CHOICE_SELECT_PROMPT

DEFAULT_SENTENCE_TRANSFORMER_MAX_LENGTH = 512
CONFIG = get_config()


class TEIServerFetchMixin:
    """
    Mixin class which contain methods to interact with TEI server.
    """

    @abstractproperty
    def action(self):
        pass

    async def _acall_api(self, json_data: dict) -> Any:
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.auth_token is not None:
            if callable(self.auth_token):
                headers["Authorization"] = self.auth_token(self.base_url)
            else:
                headers["Authorization"] = self.auth_token

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/{self.action}",
                headers=headers,
                json=json_data,
                timeout=self.timeout,
            )

        return response.json()

    def _call_api(self, json_data: dict) -> Any:
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.auth_token is not None:
            if callable(self.auth_token):
                headers["Authorization"] = self.auth_token(self.base_url)
            else:
                headers["Authorization"] = self.auth_token

        with httpx.Client() as client:
            response = client.post(
                f"{self.base_url}/{self.action}",
                headers=headers,
                json=json_data,
                timeout=self.timeout,
            )

        return response.json()


class TextEmbeddingsInferenceRerank(TEIServerFetchMixin, BaseNodePostprocessor):
    model: str = Field(default="Rerank Model", description="A dummy re-rank model name")
    base_url: str = Field(
        default="http://127.0.0.1:8082",
        description="Base URL for the rerank service.",
    )
    timeout: float = Field(
        default=60.0,
        description="Timeout in seconds for the request.",
    )
    truncate: bool = Field(
        default=False,
        description="Whether to truncate text or not.",
    )
    auth_token: Optional[Union[str, Callable[[str], str]]] = Field(
        default=None,
        description="Authentication token or authentication token generating function for authenticated requests",
    )
    top_n: int = Field(default=3, description="Number of nodes to return sorted by score.")

    @property
    def action(self) -> str:
        return "rerank"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8082",
        top_n: int = 3,
        timeout: float = 60.0,
        truncate: bool = False,
        auth_token: Optional[Union[str, Callable[[str], str]]] = None,
    ):
        super().__init__(base_url=base_url, timeout=timeout, truncate=truncate, auth_token=auth_token, top_n=top_n)

    @classmethod
    def class_name(cls: type["TextEmbeddingsInferenceRerank"]) -> str:
        return "TextEmbeddingsInferenceRerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        texts = [node.get_content(metadata_mode=MetadataMode.NONE) for node in nodes]
        json_data = {"query": query_bundle.query_str, "return_text": False, "texts": texts, "truncate": self.truncate}

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.MODEL_NAME: self.model,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:
            scores = self._call_api(json_data)

            assert len(scores) == len(nodes)

            for score in scores:
                nodes[score["index"]].score = score["score"]

            new_nodes = sorted(nodes, key=lambda x: -x.score if x.score else 0)[: self.top_n]
            event.on_end(payload={EventPayload.NODES: new_nodes})

        return new_nodes

    async def _apostprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        texts = [node.get_content(metadata_mode=MetadataMode.NONE) for node in nodes]
        json_data = {"query": query_bundle.query_str, "return_text": False, "texts": texts, "truncate": self.truncate}

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.MODEL_NAME: self.model,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:
            scores = await self._acall_api(json_data)

            assert len(scores) == len(nodes)

            for score in scores:
                nodes[score["index"]].score = score["score"]

            new_nodes = sorted(nodes, key=lambda x: -x.score if x.score else 0)[: self.top_n]
            event.on_end(payload={EventPayload.NODES: new_nodes})

        return new_nodes

    async def apostprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
        query_str: Optional[str] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        if query_str is not None and query_bundle is not None:
            raise ValueError("Cannot specify both query_str and query_bundle")
        elif query_str is not None:
            query_bundle = QueryBundle(query_str)
        else:
            pass
        return await self._apostprocess_nodes(nodes, query_bundle)


def custom_format_node_batch_fn(
    summary_nodes: List[BaseNode], node_type: Literal["Document", "Hint"] = "Document"
) -> str:
    """Default format node batch function.

    Assign each summary node a number, and format the batch of nodes.

    """
    fmt_node_txts = []
    for idx in range(len(summary_nodes)):
        number = idx + 1
        fmt_node_txts.append(
            f"{node_type} {number}:\n" f"{summary_nodes[idx].get_content(metadata_mode=MetadataMode.NONE)}"
        )
    return "\n\n".join(fmt_node_txts)


class OverridedLLMRerank(LLMRerank):
    """
    Overrided LLMRerank class.
    This enables us to include custom prompt and answer tokens on the prompt.
    """

    @staticmethod
    def overrided_parse_choice_select_answer_fn(
        answer: str, num_choices: int, raise_error: bool = False
    ) -> tuple[List[int], List[float]]:
        """Default parse choice select answer function."""
        answer_lines = answer.split("\n")
        answer_nums = []
        answer_relevances = []
        for answer_line in answer_lines:
            line_tokens = answer_line.split(":")
            if len(line_tokens) != 2:
                if not raise_error:
                    continue
                else:
                    raise ValueError(
                        f"Invalid answer line: {answer_line}. "
                        "Answer line must be of the form: "
                        "Hint <answer_num|int>: <answer_relevance|int>"
                    )
            answer_num = int(line_tokens[0].split()[1].strip())
            if answer_num > num_choices:
                continue
            answer_nums.append(answer_num)
            # extract just the first digits after the colon.
            _answer_relevance = int(line_tokens[1].strip())
            answer_relevances.append(_answer_relevance)
        return answer_nums, answer_relevances

    def __init__(
        self,
        llm: Optional[LLM] = None,
        choice_select_prompt: Optional[BasePromptTemplate] = None,
        choice_batch_size: int = 10,
        format_node_batch_fn: Optional[Callable] = None,
        parse_choice_select_answer_fn: Optional[Callable] = None,
        service_context: Optional[ServiceContext] = None,
    ) -> None:
        choice_select_prompt = choice_select_prompt or RAG_RERANK_CHOICE_SELECT_PROMPT
        if not llm:
            llm = get_rerank_llm()
        if not format_node_batch_fn:
            format_node_batch_fn = partial(custom_format_node_batch_fn, node_type="Hint")
        if not parse_choice_select_answer_fn:
            parse_choice_select_answer_fn = self.overrided_parse_choice_select_answer_fn
        super().__init__(
            llm=llm,
            choice_select_prompt=choice_select_prompt,
            choice_batch_size=choice_batch_size,
            format_node_batch_fn=format_node_batch_fn,
            parse_choice_select_answer_fn=parse_choice_select_answer_fn,
            service_context=service_context,
        )

    @classmethod
    def class_name(cls: type["OverridedLLMRerank"]) -> str:
        return "OverridedLLMRerank"

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Query bundle must be provided.")
        if len(nodes) == 0:
            return []

        initial_results: List[NodeWithScore] = []
        for idx in range(0, len(nodes), self.choice_batch_size):
            nodes_batch = [node.node for node in nodes[idx : idx + self.choice_batch_size]]

            query_str = query_bundle.query_str
            fmt_batch_str = self._format_node_batch_fn(nodes_batch)
            # call each batch independently
            raw_response = self.llm.predict(
                self.choice_select_prompt,
                context_str=fmt_batch_str,
                query_str=query_str,
            )

            raw_choices, relevances = self._parse_choice_select_answer_fn(raw_response, len(nodes_batch))
            choice_idxs = [int(choice) - 1 for choice in raw_choices]
            choice_nodes = [nodes_batch[idx] for idx in choice_idxs]
            relevances = relevances or [1.0 for _ in choice_nodes]
            initial_results.extend(
                [NodeWithScore(node=node, score=relevance) for node, relevance in zip(choice_nodes, relevances)]
            )

        # return only the nodes having the score greater than 0, ie. 1, so no need for top-k
        return [i for i in initial_results if i.score > 0]


class ReRankerType(Enum):
    DefaultLLMReRanker = auto()  # Document text used on prompt
    OverridedLLMReRanker = auto()  # Hint text used on prompt
    TEIReRanker = auto()  # ReRanker served through Text Embeddings Inference


class ReRanker:
    """
    Entry class for reranker.
    """

    def __init__(
        self,
        reranker_type: ReRankerType,
        top_n: int | None = None,
        cutoff_score: float | None = None,
    ) -> None:
        """
        Initializes appropriate re-ranker instance.
        """
        self.top_n = top_n or CONFIG.rag_facts_reranker_top_k
        self.cutoff_score = cutoff_score or CONFIG.rag_facts_reranker_cutoff_score
        if reranker_type == ReRankerType.DefaultLLMReRanker:
            self.instance = LLMRerank(
                llm=get_rerank_llm(),
                format_node_batch_fn=partial(custom_format_node_batch_fn, node_type="Document"),
                choice_select_prompt=DEFAULT_RAG_RERANK_CHOICE_SELECT_PROMPT,
                top_n=self.top_n,
            )
        elif reranker_type == ReRankerType.OverridedLLMReRanker:
            # custom LLM reranker which includes
            self.instance = OverridedLLMRerank()
        elif reranker_type == ReRankerType.TEIReRanker:
            self.instance = TextEmbeddingsInferenceRerank(
                base_url=CONFIG.rag_rerank_server_base,
                top_n=self.top_n,
            )
        else:
            raise ValueError("Invalid reranker type")

    async def _apostprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None, query_str: Optional[str] = None
    ) -> List[NodeWithScore]:
        if not query_bundle:
            query_bundle = QueryBundle(query_str=query_str)

        filtered_nodes: list[NodeWithScore] = []
        if isinstance(self.instance, TextEmbeddingsInferenceRerank):
            # apply top-n and similarity cut-off
            rearranged_nodes = await self.instance._apostprocess_nodes(nodes=nodes, query_bundle=query_bundle)
            processor = SimilarityPostprocessor(similarity_cutoff=self.cutoff_score)
            filtered_nodes = processor.postprocess_nodes(rearranged_nodes)
        elif isinstance(self.instance, OverridedLLMRerank):
            # greedily grab all the nodes having the score 1
            filtered_nodes = self.instance._postprocess_nodes(nodes=nodes, query_bundle=query_bundle)
        elif isinstance(self.instance, LLMRerank):
            # apply top-n and similarity cut-off
            rearranged_nodes = self.instance._postprocess_nodes(nodes=nodes, query_bundle=query_bundle)
            processor = SimilarityPostprocessor(similarity_cutoff=self.cutoff_score)
            filtered_nodes = processor.postprocess_nodes(rearranged_nodes)

        return filtered_nodes
