import pytest
import os
import shutil
from typing import Any
from langchain.indexes import VectorstoreIndexCreator
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.indexes.vectorstore import _get_default_text_splitter
from tests.langchain.index import ConsistentFakeEmbeddings
from heavyiq.langchain.index.heavydb.heavydb_metadata_index import HeavyDBMetadataIndex
from heavyiq.config import get_config


@pytest.fixture(scope="module")
def chroma(request: pytest.FixtureRequest) -> Chroma:
    """
    Fixture supposed to create and return log file path corresponding to the test name.
    """
    doc_a = Document(page_content="This is a table regarding movie_actress.", metadata={"source": "movie_actress"})
    doc_b = Document(page_content="This is a table regarding flights.", metadata={"source": "flights"})
    return Chroma.from_documents(
        collection_name="test_collection",
        documents=[doc_a, doc_b],
        embedding=ConsistentFakeEmbeddings(),
        ids=["doc_a", "doc_b"],
    )


@pytest.fixture(scope="module")
def heavydb_metadata_index(request: pytest.FixtureRequest, chroma: Any) -> HeavyDBMetadataIndex:
    return HeavyDBMetadataIndex(vectorstore=chroma, text_splitter=_get_default_text_splitter())  # type: ignore


@pytest.fixture(scope="module")
def vectorstore_index_creator():
    yield VectorstoreIndexCreator(
        vectorstore_kwargs={"persist_directory": "testdb"},
        embedding=ConsistentFakeEmbeddings(),
    )


@pytest.fixture(scope="function")
def testdb(tmpdir):
    """
    Create and delete metadata_index_dir on testcase setup and teardown phases.
    """
    # Create a metadata index dir
    index_path = get_config().metadata_index_dir
    os.makedirs(index_path, exist_ok=True)
    # You can perform any necessary setup here
    yield index_path
    # Teardown: Remove the directory and its contents
    shutil.rmtree(index_path)


@pytest.fixture(scope="function")
def testdb_delete_on_teardown():
    yield
    index_path = get_config().metadata_index_dir
    shutil.rmtree(index_path)


@pytest.fixture(scope="module")
def sample_table_document():
    contents = """
    Description: The movie_actress table serves to store information about the actors involved in various movies. It provides data on each actor's name, gender, their appearance order in the movie, the total number of actors in the movie, and the movie title. The primary functionality of this table is to maintain the association between movie titles and the actors who starred in them. Additionally, it provides supplementary data on the actors such as their gender and appearance order in the movie, which could be useful for analysis purposes.

    Applications: Potential use cases for the movie_actors table include:
    1. Developing a movie recommendation system based on actors and genres.
    2. Analyzing gender representation in movies and assessing the diversity in the cast.
    3. Identifying the most prolific or popular actors in specific movie genres.
    4. Investigating how appearance order (actor_order_num) correlates with an actor's prominence in the film industry.
    5. Generating trivia and quizzes for movie fans based on actor and movie relationships.

    Keywords: movie cast, film actors, actor gender, appearance order, movie database, movie titles, actor analysis, movie recommendation

    Column descriptions for movie_actors table:
    - movie_actors.movie_id: The unique identifier for a movie.
    - movie_actors.actor_name: The name of an actor in the movie.
    - movie_actors.actor_gender: The gender of the actor (Male, Female or Unspecified).
    - movie_actors.actor_order_num: The order of an actor's appearance in the credits.
    - movie_actors.num_total_actors: The total number of actors in the movie.
    - movie_actors.movie_title: The title of the movie.
    """
    yield Document(page_content=contents, metadata={"source": "movie_actress"})
