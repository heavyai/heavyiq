from abc import abstractproperty
from typing import Any, Callable, List, Optional, Union

from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle

DEFAULT_SENTENCE_TRANSFORMER_MAX_LENGTH = 512


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
