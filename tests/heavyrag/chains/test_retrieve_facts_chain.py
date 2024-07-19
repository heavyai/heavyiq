from uuid import uuid4

import pytest

from heavyrag.chains.retrieve_facts_chain import chain
from heavyrag.embed import get_chroma_client
from heavyrag.ingest import ainsert_facts

COLLECTION_NAME = "testrag"


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    # Setup code
    print("Setup: Preparing for the testcases")

    # Yield control back to the test function
    yield

    # Teardown code
    print("Tear down: Cleaning up after the testcases:")
    print(f"Deleting {COLLECTION_NAME} collection.")
    chroma_client = get_chroma_client()
    chroma_client.delete_collection(COLLECTION_NAME)


locations = [
    ("United States", "The capital city of the United States of America is Washington, D.C., known for its landmarks."),
    ("Canada", "The beautiful capital of Canada is Ottawa, located along the scenic Ottawa River."),
    ("Australia", "Canberra, the planned city, serves as the capital of Australia, with its parliamentary buildings."),
    ("Japan", "Tokyo, the bustling metropolis, is the capital of Japan and is famous for its skyscrapers."),
    ("Germany", "Berlin, rich in history, is the capital city of Germany, known for its cultural landmarks."),
    ("France", "Paris, often called the city of love, is the capital of France and home to the Eiffel Tower."),
    ("United Kingdom", "London, the historical and financial hub, is the capital of the United Kingdom."),
    ("Italy", "Rome, the eternal city, serves as the capital of Italy, known for its ancient ruins."),
    ("Brazil", "Brasília, a city planned and developed in the 1960s, is the capital of Brazil."),
    ("Russia", "Moscow, the political and economic center, is the capital of Russia and home to the Kremlin."),
    ("South Africa", "Pretoria, one of the three capitals, serves as the executive capital of South Africa."),
    ("China", "Beijing, a city with a rich history, is the capital of China and home to the Great Wall."),
    ("Mexico", "Mexico City, a bustling metropolis, is the capital of Mexico and known for its rich culture."),
    ("Argentina", "Buenos Aires, a city famous for its European-style architecture, is the capital of Argentina."),
    ("Spain", "Madrid, the vibrant and lively city, is the capital of Spain and known for its royal palace."),
    ("Saudi Arabia", "Riyadh, a city of modernity and tradition, is the capital of Saudi Arabia."),
    ("Egypt", "Cairo, a city with a long and storied history, is the capital of Egypt and home to the pyramids."),
    ("Nigeria", "Abuja, a city that was purpose-built in the 1980s, serves as the capital of Nigeria."),
    ("Indonesia", "Jakarta, a bustling and densely populated city, is the capital of Indonesia."),
    ("Turkey", "Ankara, known for its significant historical sites, serves as the capital of Turkey."),
]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "country, description",
    locations,
)
async def test_rag_retrieve_snippets(country: str, description: str):
    """
    Test heavyrag retrieval of nodes without any re-ranker.
    """
    id_ = str(uuid4())
    question = f"What is the capital of {country}?"
    snippet = description
    facts = [(id_, snippet)]
    # populate chromadb vector database
    await ainsert_facts(facts=facts, heavydb_name=COLLECTION_NAME)
    # # RAG retrieval
    chain_inputs = {
        "question": question,
        "collection_name": COLLECTION_NAME,
        "similarity_cutoff": 0.30,
        "similarity_top_k": 3,
        "use_reranker": False,
        "reranker_cutoff": 0.1,
        "reranker_top_k": 3,
    }
    retrived_snippets = await chain.ainvoke(chain_inputs)
    assert snippet in retrived_snippets


@pytest.mark.anyio
@pytest.mark.parametrize(
    "country, description",
    locations,
)
async def test_rag_retrieve_and_rerank_snippets(country: str, description: str):
    """
    Test heavyrag retrieval of nodes without any re-ranker.
    """
    id_ = str(uuid4())
    question = f"What is the capital of {country}?"
    snippet = description
    facts = [(id_, snippet)]
    # populate chromadb vector database
    await ainsert_facts(facts=facts, heavydb_name=COLLECTION_NAME)
    # # RAG retrieval
    chain_inputs = {
        "question": question,
        "collection_name": COLLECTION_NAME,
        "similarity_cutoff": 0.30,
        "similarity_top_k": 5,
        "use_reranker": True,
        "reranker_cutoff": 0.1,
        "reranker_top_k": 1,
    }
    retrived_snippets = await chain.ainvoke(chain_inputs)
    assert snippet == retrived_snippets[0]
