# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Controller class which helps to make CRUD operations on the vectorstore irrespective of it's type.
# CRUD Operations ->
#     - Snippets/facts
#       - insert
#       - list
#       - delete
#     - tables
#       - insert
#       - list
#       - delete
import threading
from abc import ABC, abstractmethod

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.indices.base import BaseIndex
from llama_index.core.schema import BaseNode, Document, NodeWithScore, TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore

from heavyiq.config import get_config
from heavyiq.langchain.heavydb import HeavyDB
from heavyrag.database import FactsModel
from heavyrag.database.database import Database as RAGDatabase
from heavyrag.embed import get_embed_model
from heavyrag.filters import get_facts_filter_matches, get_table_filter_matches
from heavyrag.loaders import aload_specific_tables, aload_table, aload_tables
from heavyrag.transform import atransform
from heavyrag.utils import convert_uuid_str_to_hex, get_nodes

CONFIG, EMBED_MODEL = get_config(), get_embed_model()


class TableAbstract(ABC):
    """
    Table abstract class which defines abstract methods for handling table node operations.
    """

    @abstractmethod
    def get_table_node(self, dbname: str, table_name: str):
        pass

    @abstractmethod
    def list_table_nodes(self, dbname: str):
        pass

    @abstractmethod
    def sync_table_nodes(self, heavydb: HeavyDB, force: bool = False):
        pass

    async def _docs_to_nodes(self, docs: list[Document]) -> list[BaseNode]:
        """
        Helps to transform a list of whole single document into multiple text nodes by using sentence splitter.
        """
        return await atransform(documents=docs)

    async def _load_table(self, *args, **kwargs) -> list[Document]:
        """
        Grab a single table info from database and then form a Document node from that.
        """
        return await aload_table(*args, **kwargs)

    async def _load_tables(self, *args, **kwargs) -> list[Document]:
        """
        Grab info of all tables from database and then form list of Document nodes from that.
        """
        return await aload_tables(*args, **kwargs)


class FactAbstract(ABC):
    """
    Table abstract class which defines abstract methods for handling fact/snippet node operations.
    """

    @abstractmethod
    def list_fact_nodes(self, dbname: str):
        pass

    @abstractmethod
    def insert_fact_nodes(self, facts: list[tuple[str, str]], dbname: str):
        pass

    @abstractmethod
    def delete_fact_nodes(self, dbname: str, fact_ids: list[str] | None = None):
        pass

    def _fact_metadata(self, dbname: str, fact_id: str, **kwargs) -> dict:
        """
        Generates fcat node's metadata.
        """
        return {"dbname": dbname, "type": "facts", "id": fact_id, **kwargs}

    def _load_facts(self, facts: list[tuple[str, str]], dbname: str) -> list[BaseNode]:
        """
        Creates TextNode's from the passed list of string tuples.
        """
        return [
            TextNode(text=fact, id_=fact_id, metadata={"dbname": dbname, "type": "facts", "id": fact_id})  # type: ignore
            for fact_id, fact in facts
        ]


class BaseController(TableAbstract, FactAbstract, ABC):
    """
    Base controller class.
    """

    @property
    @abstractmethod
    def persist_dir(self):
        """
        Vectorstore specific persist directory where all the files relevant to the vectordb resides.
        """
        pass

    @abstractmethod
    def get_vectorstore(self, *args, **kwargs):
        """
        Supposed to get the vectorstore.
        """
        pass

    def get_docstore(self) -> SimpleDocumentStore:
        """
        Get docstore.
        """
        return SimpleDocumentStore.from_persist_dir(self.persist_dir)

    def create_docstore(self) -> SimpleDocumentStore:
        """
        Create docstore.
        """
        return SimpleDocumentStore()

    def get_or_create_docstore(self) -> SimpleDocumentStore:
        """
        Get or create docstore.
        """
        try:
            return self.get_docstore()
        except FileNotFoundError:
            return self.create_docstore()

    def index(self, dbname: str) -> VectorStoreIndex:
        """
        Gets the index irrespective of the vectorstore type.
        """
        return self.get_index(self.get_vectorstore(dbname))  # type: ignore

    def get_index(self, vector_store: None | BasePydanticVectorStore) -> None | BaseIndex:
        """
        Supposed to get vectordb index.
        """
        if vector_store is None:
            return None
        docstore = self.get_docstore()
        storage_context = StorageContext.from_defaults(
            docstore=docstore, vector_store=vector_store, persist_dir=self.persist_dir
        )
        index = load_index_from_storage(storage_context=storage_context, embed_model=EMBED_MODEL)
        return index

    def get_or_create_index(self, vector_store: None | BasePydanticVectorStore) -> None | BaseIndex:
        """
        Get or create index.
        """
        try:
            index = self.get_index(vector_store)
        except FileNotFoundError:
            # index not exists so create an empty index
            docstore = self.get_or_create_docstore()
            storage_context = StorageContext.from_defaults(docstore=docstore, vector_store=vector_store)
            index = VectorStoreIndex.from_documents([], storage_context=storage_context, embed_model=EMBED_MODEL)
            index.storage_context.persist(persist_dir=self.persist_dir)
        return index

    def insert_nodes(self, index: BaseIndex, nodes: list[BaseNode]) -> None:
        """
        Helps to insert nodes into the passed index.
        """
        # insert_args get's passed to vectorstore's add method
        index.insert_nodes(nodes=nodes, show_progress=True)

    async def search_fact_nodes(self, dbname: str, question: str, top_k: int = 5) -> list[NodeWithScore]:
        """
        Helps to do search on fact nodes.
        """
        index = self.index(dbname=dbname)
        engine = index.as_retriever(
            filters=get_facts_filter_matches(dbname),
            similarity_top_k=top_k,
        )
        facts_nodes_with_score = await engine.aretrieve(question)
        return facts_nodes_with_score

    async def search_table_nodes(self, dbname: str, question: str, top_k: int = 5) -> list[NodeWithScore]:
        """
        Helps to do search on table nodes.
        """
        index = self.index(dbname=dbname)
        engine = index.as_retriever(
            filters=get_table_filter_matches(dbname),
            similarity_top_k=top_k,
        )
        facts_nodes_with_score = await engine.aretrieve(question)
        return facts_nodes_with_score

    async def sync_fact_nodes(self, ragdb: RAGDatabase, dbname: str, force: bool = False) -> None:
        """
        Synchronize specific fact nodes with facts table in the sqlite database.
        """
        vector_store = self.get_vectorstore(collection_name=dbname)
        if not vector_store:
            return None
        # list all the facts available on the rag sqlite database.
        with ragdb.get_db() as session:
            try:
                # list facts
                facts = FactsModel.list(db_session=session, heavydb_name=dbname, serialize=True)
                if not facts:
                    return None
                if force:
                    # delete all the fact nodes from index
                    await self.delete_fact_nodes(dbname=dbname)
                    # re-populate the index with the above fetched data
                    await self.insert_fact_nodes(facts=[(i["id"], i["fact"]) for i in facts], dbname=dbname)
                else:
                    # iterate over each fact and check for it's presence on the index
                    # if yes, then skip else make the node insertion
                    available_ids = [i.node_id for i in await self.list_fact_nodes(dbname=dbname)]
                    facts_to_insert = []
                    for fact in facts:
                        if fact["id"] not in available_ids:
                            facts_to_insert.append((fact["id"], fact["fact"]))
                    if facts_to_insert:
                        await self.insert_fact_nodes(facts=facts_to_insert, dbname=dbname)
            except Exception as e:
                raise e


class ChromaController(BaseController):
    """
    Controller which helps to interact with chroma vectorstore.
    """

    MAX_BATCH_SIZE = 166

    @property
    def persist_dir(self) -> str:
        return CONFIG.rag_chromadb_persist_dir

    def get_vectorstore(self, collection_name: str, metadata: dict | None = None) -> None | BasePydanticVectorStore:
        """
        Supposed to get the ChromaDB vectorstore.
        """
        from heavyrag.vector_stores.chroma import ChromaIQVectorStore

        if not EMBED_MODEL:
            return None
        return ChromaIQVectorStore(
            collection_name=collection_name,
            metadata=metadata,
        )

    def get_index_by_collection(self, collection_name: str) -> BaseIndex:
        """
        Helps to get the index from collection_name.
        """
        vector_store = self.get_vectorstore(collection_name=collection_name)
        return self.get_index(vector_store)

    async def insert_fact_nodes(self, facts: list[tuple[str, str]], dbname: str) -> None:
        """
        Create fact nodes and then insert it to the index.
        """
        vector_store = self.get_vectorstore(collection_name=dbname)
        index = self.get_or_create_index(vector_store=vector_store)
        nodes = self._load_facts(facts=facts, dbname=dbname)
        self.insert_nodes(index=index, nodes=nodes)

    async def list_fact_nodes(self, dbname: str) -> list[NodeWithScore]:
        """
        Helps to list all the fact nodes.
        """
        index = self.get_index_by_collection(collection_name=dbname)
        if not index:
            # return an empty list if in-case of no embed server
            return []
        nodes_with_score = await get_nodes(index=index, filters=get_facts_filter_matches(heavydb_name=dbname))  # type: ignore
        return nodes_with_score

    async def delete_fact_nodes(self, dbname: str, fact_ids: list[str] | None = None) -> None:
        """
        Helps to delete all the fact nodes relevant to a database/collection.
        """
        from heavyrag.ingest import adelete_database_facts, adelete_facts_by_ids

        index = self.get_index_by_collection(collection_name=dbname)
        if not index:
            return None
        if fact_ids:
            # delete only the facts associated with the passed facts ids
            await adelete_facts_by_ids(index=index, heavydb_name=dbname, facts_ids=fact_ids)  # type: ignore
        else:
            # delete all the facts relevant to a particular database
            await adelete_database_facts(index=index, heavydb_name=dbname)  # type: ignore

    async def get_table_node(self, dbname: str, table_name: str) -> NodeWithScore | None:
        """
        Gets the relevant table node.
        """
        index = self.get_index_by_collection(collection_name=dbname)
        nodes_with_score = await get_nodes(index=index, filters=get_table_filter_matches(heavydb_name=dbname, table_name=table_name))  # type: ignore
        if nodes_with_score:
            return nodes_with_score[0]
        return None

    async def list_table_nodes(self, dbname: str) -> list[NodeWithScore]:
        """
        List all table nodes specific to a particular database.
        """
        index = self.get_index_by_collection(collection_name=dbname)
        if not index:
            return []
        nodes_with_score = await get_nodes(index=index, filters=get_table_filter_matches(heavydb_name=dbname))  # type: ignore
        return nodes_with_score

    def insert_nodes_in_batches(self, index: VectorStoreIndex, nodes: list[BaseNode], show_progress: bool = True):
        """
        Batch Insertion in chormadb should not exceed a certain limit. ie. not more than 166 nodes.
        """
        for i in range(0, len(nodes), self.MAX_BATCH_SIZE):
            batch = nodes[i : i + self.MAX_BATCH_SIZE]
            index.insert_nodes(nodes=batch, show_progress=show_progress)

    async def sync_table_nodes(self, heavydb: HeavyDB, force: bool = False) -> None:
        """
        Synchronize nodes (documents or text content) with a table in the database.

        Parameters:
            - dbname (str): The name of the database where the nodes are being synchronized.
            - force (bool, optional): If set to True, the function will force a resynchronization of the nodes
                    even if they are already up to date. Defaults to False, meaning that
                    synchronization will only occur if necessary.
        """
        from heavyrag.ingest import delete_all_table_nodes, delete_nodes_by_ids

        dbname = heavydb._dbname
        index = self.get_or_create_index(vector_store=self.get_vectorstore(collection_name=dbname))
        if not index:
            return None
        if force:
            # delete and recreate all the table nodes relevant to a database
            await delete_all_table_nodes(index=index, dbname=dbname)  # type: ignore
            docs = await self._load_tables(heavydb)
            nodes = await self._docs_to_nodes(docs)
            self.insert_nodes_in_batches(index, nodes, show_progress=True)
            return None

        all_tables = heavydb.get_usable_table_names()
        available_tables: dict[str, str] = {}  # store table_name as key and node_id as value
        for i in await self.list_table_nodes(dbname=dbname):
            available_tables[i.metadata["name"]] = i.node_id

        # remove deleted tables from the index
        tables_to_delete = set(available_tables.keys()) - set(all_tables)
        tables_to_insert = set(all_tables) - set(available_tables.keys())

        ids_to_delete = [available_tables[i] for i in tables_to_delete]

        if ids_to_delete:
            delete_nodes_by_ids(index=index, ids=ids_to_delete)  # type: ignore

        # insert the missing tables
        if tables_to_insert:
            docs = await aload_specific_tables(heavydb=heavydb, tables=tables_to_insert)
            nodes = await self._docs_to_nodes(docs)
            self.insert_nodes_in_batches(index, nodes, show_progress=True)
        return None


class FaissController(BaseController):
    """
    Controller which helps to interact with faiss vectorstore.
    """

    _cached_vector_store = None
    _lock = threading.Lock()

    @property
    def persist_dir(self) -> str:
        return CONFIG.rag_faiss_persist_dir

    def get_vectorstore(self, *args, **kwargs) -> None | BasePydanticVectorStore:
        """
        Supposed to get the Faiss vectorstore.
        """
        from heavyrag.vector_stores.faiss import FaissIQVectorStore

        # cache this vs initialisation so that it won't be readed again and again
        if self._cached_vector_store:
            return self._cached_vector_store

        if not EMBED_MODEL:
            return None

        # Thread-safe initialization with lock
        with self._lock:
            if self._cached_vector_store is None:
                self._cached_vector_store = FaissIQVectorStore(
                    persist_dir=self.persist_dir, dimension=CONFIG.rag_embed_dimension
                )  # todo: set the dimension here

        return self._cached_vector_store

    async def insert_fact_nodes(self, facts: list[tuple[str, str]], dbname: str) -> None:
        """
        Create fact nodes and then insert it to the index.
        """
        vector_store = self.get_vectorstore()
        index = self.get_or_create_index(vector_store=vector_store)
        nodes = self._load_facts(facts=facts, dbname=dbname)
        self.insert_nodes(index=index, nodes=nodes)

    async def list_fact_nodes(self, dbname: str) -> list[NodeWithScore]:
        """
        Helps to list all the fact nodes.
        """
        vector_store = self.get_vectorstore()
        index = self.get_or_create_index(vector_store=vector_store)
        nodes_with_score = await get_nodes(index=index, filters=get_facts_filter_matches(heavydb_name=dbname))  # type: ignore
        return nodes_with_score

    async def _delete_facts_by_ids(self, index: VectorStoreIndex, dbname: str, fact_ids: list[str]):
        """
        Helps to delete facts by fact ids.
        """
        nodes_with_score = await get_nodes(index=index, filters=get_facts_filter_matches(heavydb_name=dbname))
        node_ids_to_delete = [i.node_id for i in nodes_with_score if i.node_id in fact_ids]
        if node_ids_to_delete:
            index.vector_store.remove(node_ids=node_ids_to_delete)  # type: ignore

    async def _delete_database_facts(self, index: VectorStoreIndex, dbname: str):
        """
        Delete all facts relevant to a database.
        """
        nodes_with_score = await get_nodes(index=index, filters=get_facts_filter_matches(heavydb_name=dbname))
        node_ids_to_delete = [i.node_id for i in nodes_with_score]
        if node_ids_to_delete:
            index.vector_store.remove(node_ids=node_ids_to_delete)  # type: ignore

    async def delete_fact_nodes(self, dbname: str, fact_ids: list[str] | None = None) -> None:
        """
        Helps to delete all the fact nodes relevant to a database/collection.
        """
        index = self.get_or_create_index(vector_store=self.get_vectorstore())
        if fact_ids:
            # delete only the facts associated with the passed facts ids
            await self._delete_facts_by_ids(index=index, dbname=dbname, fact_ids=fact_ids)  # type: ignore
        else:
            # delete all the facts relevant to a particular database
            await self._delete_database_facts(index=index, dbname=dbname)  # type: ignore

    async def get_table_node(self, dbname: str, table_name: str) -> NodeWithScore | None:
        """
        Gets the relevant table node.
        """
        index = self.get_or_create_index(vector_store=self.get_vectorstore())
        nodes_with_score = await get_nodes(index=index, filters=get_table_filter_matches(heavydb_name=dbname, table_name=table_name))  # type: ignore
        if nodes_with_score:
            return nodes_with_score[0]
        return None

    async def _docs_to_nodes(self, docs: list[Document]) -> list[BaseNode]:
        """
        Helps to transform a list of whole single document into multiple text nodes by using sentence splitter.
        """
        nodes = await atransform(documents=docs)
        # alway keep hex codes as node id
        for node in nodes:
            node_id = node.node_id
            if "-" in node_id:
                node.id_ = convert_uuid_str_to_hex(node_id)
        return nodes

    async def list_table_nodes(self, dbname: str) -> list[NodeWithScore]:
        """
        List all table nodes specific to a particular database.
        """
        index = self.get_or_create_index(vector_store=self.get_vectorstore())
        nodes_with_score = await get_nodes(index=index, filters=get_table_filter_matches(heavydb_name=dbname))  # type: ignore
        return nodes_with_score

    async def _delete_all_table_nodes(self, index: BaseIndex, dbname: str) -> None:
        """
        Delete all table nodes.
        """
        nodes_with_score = await get_nodes(index=index, filters=get_table_filter_matches(heavydb_name=dbname))  # type: ignore
        node_ids_to_delete = [i.node_id for i in nodes_with_score]
        if node_ids_to_delete:
            index.vector_store.remove(node_ids=node_ids_to_delete)  # type: ignore

    async def sync_table_nodes(self, heavydb: HeavyDB, force: bool = False) -> None:
        """
        Synchronize nodes (documents or text content) with a table in the database.

        Parameters:
            - dbname (str): The name of the database where the nodes are being synchronized.
            - force (bool, optional): If set to True, the function will force a resynchronization of the nodes
                    even if they are already up to date. Defaults to False, meaning that
                    synchronization will only occur if necessary.
        """
        dbname = heavydb._dbname
        index = self.get_or_create_index(vector_store=self.get_vectorstore())
        if not index:
            return None
        if force:
            # delete and recreate all the table nodes relevant to a database
            await self._delete_all_table_nodes(index=index, dbname=dbname)  # type: ignore
            docs = await self._load_tables(heavydb)
            nodes = await self._docs_to_nodes(docs)
            index.insert_nodes(nodes=nodes, show_progress=True)
            return None

        all_tables = heavydb.get_usable_table_names()
        available_tables: dict[str, str] = {}  # store table_name as key and node_id as value
        for i in await self.list_table_nodes(dbname=dbname):
            available_tables[i.metadata["name"]] = i.node_id

        # remove deleted tables from the index
        tables_to_delete = set(available_tables.keys()) - set(all_tables)
        tables_to_insert = set(all_tables) - set(available_tables.keys())

        ids_to_delete = [available_tables[i] for i in tables_to_delete]

        if ids_to_delete:
            index.vector_store.remove(node_ids=ids_to_delete)  # type: ignore

        # insert the missing tables
        if tables_to_insert:
            docs = await aload_specific_tables(heavydb=heavydb, tables=tables_to_insert)
            nodes = await self._docs_to_nodes(docs)
            index.insert_nodes(nodes=nodes, show_progress=True)
        return None


# Step 4: Method to choose the appropriate controller based on a string argument
def get_controller(controller_type: str) -> BaseController:
    if controller_type.lower() == "chroma":
        return ChromaController()
    elif controller_type.lower() == "faiss":
        return FaissController()
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")


rag_controller = get_controller(CONFIG.rag_vectordb_type)
