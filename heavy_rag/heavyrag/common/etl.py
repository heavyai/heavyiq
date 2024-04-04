from abc import ABCMeta, abstractmethod, abstractproperty
from functools import cached_property

from llama_index.core import VectorStoreIndex
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document, NodeWithScore


class BaseETL(metaclass=ABCMeta):
    """
    Base class for all ETL classes.
    """

    INDEX_ON_PROGRESS: dict[str, bool] = {}

    def __init__(self, reader_cls: type[BaseReader], reader_init_kwargs: dict | None = None):
        self.reader_cls = reader_cls
        self.reader_init_kwargs = reader_init_kwargs or {}

    @cached_property
    def reader(self):
        """
        Returns the reader instance.
        """
        return self.reader_cls(**self.reader_init_kwargs)

    @abstractmethod
    def read(self) -> list[Document]:
        """
        Read documents for building index from any sources like folders or databases using BaseReader instance.
        """
        pass

    @abstractmethod
    def sync_index(self, collection_name: str) -> VectorStoreIndex:
        """
        Helps to sync index with the latest documents/nodes.
        """
        pass

    def index(self, collection_name) -> VectorStoreIndex:
        """
        Get or create or update VectorStore Index from the collection name.
        """
        if collection_name in self.INDEX_ON_PROGRESS:
            raise ValueError(f"Index building in-progress for {collection_name}, please wait.")

        try:
            self.INDEX_ON_PROGRESS[collection_name] = True
            index = self.sync_index(collection_name)
        except Exception as e:
            raise e
        finally:
            self.INDEX_ON_PROGRESS.pop(collection_name, None)

        return index

    @abstractmethod
    def retrieve_index(self, index: VectorStoreIndex, question: str) -> list[NodeWithScore]:
        """
        Retrieves relevant nodes from the index.
        """
        pass

    @abstractmethod
    def run(self, question: str) -> list[NodeWithScore]:
        pass
