import argparse
import asyncio
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Literal

import chromadb.api
import chromadb.api.client
import chromadb.config
import fitz
import nest_asyncio
from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.extractors import QuestionsAnsweredExtractor, SummaryExtractor, TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.utils import get_tqdm_iterable
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.langchain import LangChainLLM
from llama_index.readers.web.simple_web.base import SimpleWebPageReader
from llama_index.vector_stores.chroma import ChromaVectorStore

import chromadb
from heavyiq.langchain.llms import LLMType, get_llm_by_type
from heavyiq.logging_utils import get_heavyiq_logger

nest_asyncio.apply()

RAW_LLM = get_llm_by_type(LLMType.DEFAULT)
LLM = LangChainLLM(llm=RAW_LLM)
node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=128, paragraph_separator="\n\n")
logger = get_heavyiq_logger()

COLLECTION_NAME = "documents"
file_path = "sources/rag_documents/flight_1/heavydb.pdf"
pickle_file_path = "pickle/"
doc = fitz.open(file_path)
CHROMA_PERSIST_DIR = "chromadb"

QUESTION_GEN_USER_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "generate the relevant questions."
)

QUESTION_GEN_SYS_TMPL = """You are a Teacher/ Professor. Your task is to setup \
{num_questions} questions for an upcoming \
quiz/examination. The questions should be diverse in nature \
across the document. Restrict the questions to the \
context information provided."""

# QUESTION_TMPL = """Here is the context: # use for llam3
# {context_str}

# Given the contextual information, \
# generate {num_questions} questions this context can provide \
# specific answers to which are unlikely to be found elsewhere.

# Higher-level summaries of surrounding context may be provided \
# as well. Try using these summaries to generate better questions \
# that this context can answer."""

QUESTION_TMPL = """Here is the context:
{context_str}

Given the contextual information, \
generate {num_questions} question and answer pairs that this context can provide \
specific answers to which are unlikely to be found elsewhere.

Higher-level summaries of surrounding context may be provided \
as well. Try using these summaries to generate better questions \
that this context can answer."""


LLAMA3_USER_TMPL = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

PHI3_USER_TMPL = """<|user|>
{user_message}<|end|>
<|assistant|>
"""
TEMPLATE_TO_USE = PHI3_USER_TMPL
QUESTION_PROMPT_TMPL = TEMPLATE_TO_USE.format(user_message=QUESTION_TMPL)


DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)
TEXT_QA_PROMPT_TMPL = TEMPLATE_TO_USE.format(user_message=DEFAULT_TEXT_QA_PROMPT_TMPL)
DEFAULT_TEXT_QA_PROMPT = PromptTemplate(TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER)

#####################
# INDEX
#####################


def get_hf_embeddings() -> BaseEmbedding:
    return HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_chromadb_client(persist_dir: str) -> chromadb.api.client.Client:
    """
    Gets the chromadb client.
    """
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=chromadb.config.Settings(anonymized_telemetry=False, is_persistent=True),
    )


def get_vectorstore(collection_name: str) -> ChromaVectorStore:
    """
    Get or Create a VectorStore.
    """
    chroma_client = get_chromadb_client(CHROMA_PERSIST_DIR)
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    # set up ChromaVectorStore
    return ChromaVectorStore.from_collection(chroma_collection)


def get_or_create_index(collection_name: str, use_async: bool = False) -> VectorStoreIndex:
    """
    Get or creates new index from user uploaded documents.
    """
    vector_store = get_vectorstore(
        collection_name,
    )
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        show_progress=True,
        use_async=use_async,
        insert_batch_size=166,
        embed_model=get_hf_embeddings(),
    )
    return index


############################
# Loaders
############################


def read_pdf(file_path: str) -> list[BaseNode]:
    """
    Read a loacl PDF file, split it into chunks.
    """
    logger.debug(f"Started reading PDF file {file_path}")
    doc = fitz.open(file_path)
    text_chunks, doc_idxs = [], []
    split_with_progress = get_tqdm_iterable(doc, True, "Parsing Each Page")
    for doc_idx, page in enumerate(split_with_progress):
        page_text = page.get_text("text")
        cur_text_chunks = node_parser.split_text(page_text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))
    logger.debug("Splitted into chunks.")

    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(text=text_chunk, metadata={"page_no": doc_idxs[idx] + 1})
        # src_page = doc[src_doc_idx]
        nodes.append(node)
    return nodes


def read_url(url: str) -> list[BaseNode]:
    """
    Reads a web page, converts it to text, splits into chunks.
    """
    reader = SimpleWebPageReader(html_to_text=True)
    docs = reader.load_data(urls=[url])
    nodes = node_parser.get_nodes_from_documents(docs)
    return nodes


async def transform(nodes: list[BaseNode]) -> list[BaseNode]:
    extractors = [
        QuestionsAnsweredExtractor(
            questions=3,
            llm=LLM,
            prompt_template=QUESTION_PROMPT_TMPL,
            metadata_mode=MetadataMode.EMBED,
            num_workers=8,
            show_progress=True,
        ),
        get_hf_embeddings(),
    ]
    pipeline = IngestionPipeline(transformations=[*extractors])
    start = time.time()
    splitted_nodes = await pipeline.arun(nodes=nodes, in_place=False, show_progress=True)
    end = time.time()
    time_taken = end - start
    print(f"Time taken for processing {(len(nodes))} nodes: {time_taken}")
    return splitted_nodes


def ingest(splitted_nodes: list[BaseNode], index: VectorStoreIndex | None = None) -> VectorStoreIndex:
    """
    Helps to insert nodes to existing index.
    """
    if not index:
        index = get_or_create_index(COLLECTION_NAME, use_async=True)
    index.build_index_from_nodes(splitted_nodes)
    return index


def get_pickle_path(file_path: str) -> Path:
    path = Path(file_path)
    return path.parent / (".".join(path.name.split(".")[:-1]) + ".pickle")


def is_pickle_path_exists(file_path: str) -> bool:
    pickle_path = get_pickle_path(file_path)
    return pickle_path.resolve().exists()


def pickle_nodes(file_path: str, nodes: list[BaseNode]):
    pickle_path = get_pickle_path(file_path)
    with open(pickle_path, "wb") as f:
        pickle.dump(nodes, f, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_nodes(file_path: str) -> list[BaseNode] | None:
    pickle_path = get_pickle_path(file_path)
    if not pickle_path.resolve().exists():
        return None
    with open(pickle_path, "rb") as f:
        nodes = pickle.load(f)
    return nodes


async def insert(file_path: str):
    if is_pickle_path_exists(file_path):
        logger.info("Unpacking nodes from pickle file.")
        splitted_nodes = unpickle_nodes(file_path)
    else:
        nodes = read_pdf(file_path)
        splitted_nodes = await transform(nodes)
        pickle_nodes(file_path, splitted_nodes)
    with open("out.txt", "w") as w:
        txt = ""
        for n in splitted_nodes:
            txt += n.get_content(metadata_mode=MetadataMode.ALL)
            txt += "=" * 200
        w.write(txt)
    print("Started ingesting nodes to vectorDB")
    ingest(splitted_nodes)
    print("successfully added nodes to index.")


async def ask(file_path: str, question: str):
    if not is_pickle_path_exists(file_path):
        print("Do insert before asking...")
        sys.exit(0)
    index = get_or_create_index(COLLECTION_NAME, use_async=True)
    engine = index.as_query_engine(
        llm=LLM,
        similarity_top_k=1,
        response_mode=ResponseMode.SIMPLE_SUMMARIZE,
        text_qa_template=DEFAULT_TEXT_QA_PROMPT,
    )
    response = engine.query(question)
    print(response.response)


async def main(file_path: str, question: str | None = None, action: Literal["ask", "insert"] = "ask"):
    out = None
    if action == "ask":
        assert question
        out = await ask(file_path, question)
    elif action == "insert":
        out = await insert(file_path)
    if out:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Documents RAG")

    parser.add_argument("action", help="insert or ask")
    parser.add_argument("path", help="File Path")
    parser.add_argument("--question", help="Question", default=None)

    args = parser.parse_args()
    asyncio.run(main(args.path, action=args.action, question=args.question))
