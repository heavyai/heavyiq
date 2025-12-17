"""
Tests for FAISS vector store.

Tests the FaissIQVectorStore class and related functionality.
"""

import multiprocessing
import os
import shutil
import tempfile
import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch
from uuid import uuid4

import faiss
import numpy as np
import pytest

from heavyiq.config import HeavyIQConfig


def generate_hex_id() -> str:
    """Generate a valid hex UUID for FAISS node IDs."""
    return uuid4().hex


def _multiprocess_worker(persist_dir: str, dimension: int, worker_id: int, result_queue: multiprocessing.Queue):
    """
    Worker function for multiprocess FAISS tests.
    Each worker creates its own store and adds nodes.
    """
    try:
        import heavyrag.vector_stores.faiss as faiss_module
        from heavyrag.vector_stores.faiss import FaissIQVectorStore
        from llama_index.core.schema import TextNode
        
        # Reset global state for this process so FaissIQVectorStore creates fresh index
        faiss_module.faiss_index = None
        faiss_module.last_update_time = 0
        
        # FaissIQVectorStore will create the index internally via get_faiss_index()
        store = FaissIQVectorStore(
            persist_dir=persist_dir,
            dimension=dimension,
        )
        
        added_count = 0
        for i in range(5):
            node = TextNode(
                text=f"Worker {worker_id} doc {i}",
                id_=generate_hex_id(),
                embedding=np.random.rand(dimension).tolist(),
                metadata={"worker": worker_id, "doc": i},
            )
            store.add([node])
            added_count += 1
        
        result_queue.put({"worker_id": worker_id, "added": added_count, "error": None})
    except Exception as e:
        result_queue.put({"worker_id": worker_id, "added": 0, "error": str(e)})


class TestFaissVectorStore:
    """Tests for FaissIQVectorStore."""

    @pytest.fixture
    def temp_persist_dir(self):
        """Create a temporary directory for FAISS persistence."""
        temp_dir = tempfile.mkdtemp(prefix="faiss_test_")
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_config(self, temp_persist_dir):
        """Create a mock config for testing."""
        config = MagicMock(spec=HeavyIQConfig)
        config.rag_faiss_persist_dir = temp_persist_dir
        config.rag_embed_dimension = 384  # Common embedding dimension
        config.rag_vectordb_type = "faiss"
        return config

    @pytest.fixture
    def faiss_vectorstore(self, mock_config, temp_persist_dir):
        """Create a FaissIQVectorStore instance for testing."""
        import heavyrag.vector_stores.faiss as faiss_module
        from heavyrag.vector_stores.faiss import FaissIQVectorStore

        # Reset global state so FaissIQVectorStore creates a fresh index
        faiss_module.faiss_index = None
        faiss_module.last_update_time = 0

        store = FaissIQVectorStore(
            persist_dir=temp_persist_dir,
            dimension=mock_config.rag_embed_dimension,
        )
        yield store
        
        # Cleanup global state
        faiss_module.faiss_index = None

    def test_init_creates_persist_dir(self, mock_config, temp_persist_dir):
        """Test that initialization creates the persist directory."""
        import heavyrag.vector_stores.faiss as faiss_module
        from heavyrag.vector_stores.faiss import FaissIQVectorStore

        # Remove the temp dir first
        shutil.rmtree(temp_persist_dir)
        assert not os.path.exists(temp_persist_dir)

        # Reset global state
        faiss_module.faiss_index = None

        store = FaissIQVectorStore(
            persist_dir=temp_persist_dir,
            dimension=mock_config.rag_embed_dimension,
        )

        assert os.path.exists(temp_persist_dir)

    def test_add_nodes(self, faiss_vectorstore, mock_config):
        """Test adding nodes to the vector store."""
        from llama_index.core.schema import TextNode

        # Create test nodes with embeddings (using valid hex UUIDs)
        dimension = mock_config.rag_embed_dimension
        node1_id = generate_hex_id()
        node2_id = generate_hex_id()
        nodes = [
            TextNode(
                text="Test document 1",
                id_=node1_id,
                embedding=np.random.rand(dimension).tolist(),
                metadata={"source": "test1"},
            ),
            TextNode(
                text="Test document 2",
                id_=node2_id,
                embedding=np.random.rand(dimension).tolist(),
                metadata={"source": "test2"},
            ),
        ]

        # Add nodes
        result = faiss_vectorstore.add(nodes)

        assert len(result) == 2

    def test_persist_and_load(self, mock_config, temp_persist_dir):
        """Test persisting and loading the index."""
        import heavyrag.vector_stores.faiss as faiss_module
        from heavyrag.vector_stores.faiss import FaissIQVectorStore
        from llama_index.core.schema import TextNode

        # Reset global state
        faiss_module.faiss_index = None

        dimension = mock_config.rag_embed_dimension
        store = FaissIQVectorStore(
            persist_dir=temp_persist_dir,
            dimension=dimension,
        )

        # Add a node
        test_node_id = generate_hex_id()
        node = TextNode(
            text="Test document",
            id_=test_node_id,
            embedding=np.random.rand(dimension).tolist(),
            metadata={"key": "value"},
        )
        store.add([node])

        # Persist
        store.persist()

        # Check files exist
        assert os.path.exists(os.path.join(temp_persist_dir, "faiss_index.bin"))
        assert os.path.exists(os.path.join(temp_persist_dir, "metadata.pkl"))

    def test_delete_nodes(self, faiss_vectorstore, mock_config):
        """Test deleting nodes from the vector store."""
        from llama_index.core.schema import TextNode

        dimension = mock_config.rag_embed_dimension

        # Add nodes first (using valid hex UUIDs)
        delete_node1_id = generate_hex_id()
        delete_node2_id = generate_hex_id()
        nodes = [
            TextNode(
                text="Test document 1",
                id_=delete_node1_id,
                embedding=np.random.rand(dimension).tolist(),
                metadata={"source": "test1"},
            ),
            TextNode(
                text="Test document 2",
                id_=delete_node2_id,
                embedding=np.random.rand(dimension).tolist(),
                metadata={"source": "test2"},
            ),
        ]
        faiss_vectorstore.add(nodes)

        # Delete one node using FAISS remove method
        faiss_vectorstore.remove(node_ids=[delete_node1_id])

        # The node should be removed from metadata
        from heavyrag.utils import uuid4_hex_to_int64
        node1_int_id = uuid4_hex_to_int64(delete_node1_id)
        assert node1_int_id not in faiss_vectorstore._metadata_store

    def test_query(self, faiss_vectorstore, mock_config):
        """Test querying the vector store."""
        from llama_index.core.schema import TextNode
        from llama_index.core.vector_stores.types import VectorStoreQuery

        dimension = mock_config.rag_embed_dimension

        # Add nodes (using valid hex UUIDs)
        python_node_id = generate_hex_id()
        js_node_id = generate_hex_id()
        nodes = [
            TextNode(
                text="Python is a programming language",
                id_=python_node_id,
                embedding=np.random.rand(dimension).tolist(),
                metadata={"topic": "programming"},
            ),
            TextNode(
                text="JavaScript is also a programming language",
                id_=js_node_id,
                embedding=np.random.rand(dimension).tolist(),
                metadata={"topic": "programming"},
            ),
        ]
        faiss_vectorstore.add(nodes)

        # Query
        query_embedding = np.random.rand(dimension).tolist()
        query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=2,
        )

        result = faiss_vectorstore.query(query)

        assert result is not None
        assert hasattr(result, "nodes")
        assert hasattr(result, "similarities")

    def test_reset(self, faiss_vectorstore, mock_config):
        """Test resetting the vector store."""
        from llama_index.core.schema import TextNode

        dimension = mock_config.rag_embed_dimension

        # Add a node (using valid hex UUID)
        reset_node_id = generate_hex_id()
        node = TextNode(
            text="Test document",
            id_=reset_node_id,
            embedding=np.random.rand(dimension).tolist(),
            metadata={"key": "value"},
        )
        faiss_vectorstore.add([node])

        # Verify node exists
        assert len(faiss_vectorstore._metadata_store) > 0

        # Reset
        faiss_vectorstore.reset()

        # Verify cleared
        assert len(faiss_vectorstore._metadata_store) == 0

    def test_get_ids(self, faiss_vectorstore, mock_config):
        """Test getting all IDs from the vector store."""
        from llama_index.core.schema import TextNode

        dimension = mock_config.rag_embed_dimension

        # Add nodes (using valid hex UUIDs)
        id1 = generate_hex_id()
        id2 = generate_hex_id()
        nodes = [
            TextNode(
                text="Test document 1",
                id_=id1,
                embedding=np.random.rand(dimension).tolist(),
            ),
            TextNode(
                text="Test document 2",
                id_=id2,
                embedding=np.random.rand(dimension).tolist(),
            ),
        ]
        faiss_vectorstore.add(nodes)

        ids = faiss_vectorstore.get_ids()

        assert len(ids) >= 2

    def test_class_name(self, faiss_vectorstore):
        """Test class_name class method."""
        assert faiss_vectorstore.class_name() == "FaissIQVectorStore"


class TestFaissIndex:
    """Tests for FAISS index management functions."""

    @pytest.fixture
    def temp_persist_dir(self):
        """Create a temporary directory for FAISS persistence."""
        temp_dir = tempfile.mkdtemp(prefix="faiss_index_test_")
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_get_faiss_index_creates_new(self, temp_persist_dir):
        """Test get_faiss_index creates a new index when none exists."""
        # Reset global faiss_index
        import heavyrag.vector_stores.faiss as faiss_module

        original_index = faiss_module.faiss_index
        faiss_module.faiss_index = None

        try:
            from heavyrag.vector_stores.faiss import get_faiss_index

            index = get_faiss_index(persist_dir=temp_persist_dir, dimension=128)

            assert index is not None
        finally:
            faiss_module.faiss_index = original_index

    def test_get_faiss_index_loads_existing(self, temp_persist_dir):
        """Test get_faiss_index loads an existing index from disk."""
        import heavyrag.vector_stores.faiss as faiss_module

        original_index = faiss_module.faiss_index
        faiss_module.faiss_index = None

        try:
            # Create and save an index
            dimension = 128
            index = faiss.IndexFlatL2(dimension)

            # Add some vectors
            vectors = np.random.rand(5, dimension).astype("float32")
            index.add(vectors)

            index_path = os.path.join(temp_persist_dir, "faiss_index.bin")
            faiss.write_index(index, index_path)

            # Now load it
            from heavyrag.vector_stores.faiss import get_faiss_index

            loaded_index = get_faiss_index(persist_dir=temp_persist_dir, dimension=dimension)

            assert loaded_index is not None
            assert loaded_index.ntotal == 5  # Should have 5 vectors
        finally:
            faiss_module.faiss_index = original_index

    def test_get_faiss_index_cached(self, temp_persist_dir):
        """Test get_faiss_index returns cached index on subsequent calls."""
        import heavyrag.vector_stores.faiss as faiss_module

        original_index = faiss_module.faiss_index
        faiss_module.faiss_index = None

        try:
            from heavyrag.vector_stores.faiss import get_faiss_index

            index1 = get_faiss_index(persist_dir=temp_persist_dir, dimension=128)
            index2 = get_faiss_index(persist_dir=temp_persist_dir, dimension=128)

            # Should be the same cached instance
            assert index1 is index2
        finally:
            faiss_module.faiss_index = original_index


class TestFaissFileLock:
    """Tests for FAISS file-based locking mechanism."""

    @pytest.fixture
    def temp_persist_dir(self):
        """Create a temporary directory for FAISS persistence."""
        temp_dir = tempfile.mkdtemp(prefix="faiss_lock_test_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_file_lock_creates_lock_file(self, temp_persist_dir):
        """Test that FaissFileLock creates the lock file."""
        from heavyrag.vector_stores.faiss import FaissFileLock

        lock_file = os.path.join(temp_persist_dir, ".faiss.lock")
        file_lock = FaissFileLock(lock_file)

        with file_lock.write_lock():
            assert os.path.exists(lock_file)

    def test_read_lock_allows_multiple_readers(self, temp_persist_dir):
        """Test that multiple read locks can be acquired simultaneously."""
        from heavyrag.vector_stores.faiss import FaissFileLock
        import threading
        import time

        lock_file = os.path.join(temp_persist_dir, ".faiss.lock")
        file_lock = FaissFileLock(lock_file, timeout=5.0)

        results = []
        errors = []

        def reader(reader_id):
            try:
                with file_lock.read_lock():
                    results.append(f"reader_{reader_id}_acquired")
                    time.sleep(0.1)  # Hold lock briefly
                    results.append(f"reader_{reader_id}_released")
            except Exception as e:
                errors.append(str(e))

        # Start multiple readers simultaneously
        threads = [threading.Thread(target=reader, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        # All readers should have acquired the lock
        assert sum(1 for r in results if "acquired" in r) == 3

    def test_write_lock_blocks_other_writers(self, temp_persist_dir):
        """Test that write lock blocks other writers."""
        from heavyrag.vector_stores.faiss import FaissFileLock
        import threading
        import time

        lock_file = os.path.join(temp_persist_dir, ".faiss.lock")
        file_lock = FaissFileLock(lock_file, timeout=5.0)

        sequence = []
        lock_obj = threading.Lock()

        def writer(writer_id, hold_time):
            with file_lock.write_lock():
                with lock_obj:
                    sequence.append(f"writer_{writer_id}_start")
                time.sleep(hold_time)
                with lock_obj:
                    sequence.append(f"writer_{writer_id}_end")

        # Writer 1 holds for longer, Writer 2 starts after a small delay
        t1 = threading.Thread(target=writer, args=(1, 0.3))
        t2 = threading.Thread(target=writer, args=(2, 0.1))

        t1.start()
        time.sleep(0.05)  # Give t1 time to acquire lock
        t2.start()

        t1.join()
        t2.join()

        # Writer 2 should not start until Writer 1 ends
        writer1_end_idx = sequence.index("writer_1_end")
        writer2_start_idx = sequence.index("writer_2_start")
        assert writer2_start_idx > writer1_end_idx, f"Writer 2 started before Writer 1 ended: {sequence}"

    def test_write_lock_blocks_readers(self, temp_persist_dir):
        """Test that write lock blocks readers."""
        from heavyrag.vector_stores.faiss import FaissFileLock
        import threading
        import time

        lock_file = os.path.join(temp_persist_dir, ".faiss.lock")
        file_lock = FaissFileLock(lock_file, timeout=5.0)

        sequence = []
        lock_obj = threading.Lock()

        def writer():
            with file_lock.write_lock():
                with lock_obj:
                    sequence.append("writer_start")
                time.sleep(0.3)
                with lock_obj:
                    sequence.append("writer_end")

        def reader():
            with file_lock.read_lock():
                with lock_obj:
                    sequence.append("reader_start")
                time.sleep(0.1)
                with lock_obj:
                    sequence.append("reader_end")

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)

        t1.start()
        time.sleep(0.05)  # Give writer time to acquire lock
        t2.start()

        t1.join()
        t2.join()

        # Reader should not start until writer ends
        writer_end_idx = sequence.index("writer_end")
        reader_start_idx = sequence.index("reader_start")
        assert reader_start_idx > writer_end_idx, f"Reader started before writer ended: {sequence}"

    def test_lock_timeout(self, temp_persist_dir):
        """Test that lock acquisition times out properly."""
        from heavyrag.vector_stores.faiss import FaissFileLock
        import threading
        import time

        lock_file = os.path.join(temp_persist_dir, ".faiss.lock")
        file_lock = FaissFileLock(lock_file, timeout=0.5)  # Short timeout

        timeout_occurred = threading.Event()

        def holder():
            with file_lock.write_lock():
                time.sleep(2.0)  # Hold lock longer than timeout

        def waiter():
            try:
                with file_lock.write_lock():
                    pass  # Should not reach here
            except TimeoutError:
                timeout_occurred.set()

        t1 = threading.Thread(target=holder)
        t2 = threading.Thread(target=waiter)

        t1.start()
        time.sleep(0.1)  # Give holder time to acquire
        t2.start()

        t2.join(timeout=3.0)  # Wait for waiter to timeout
        t1.join(timeout=3.0)

        assert timeout_occurred.is_set(), "Expected timeout did not occur"


class TestFaissConcurrentWrites:
    """Tests for concurrent FAISS writes with file locking."""

    @pytest.fixture
    def temp_persist_dir(self):
        """Create a temporary directory for FAISS persistence."""
        temp_dir = tempfile.mkdtemp(prefix="faiss_concurrent_test_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_config(self, temp_persist_dir):
        """Create a mock config for testing."""
        config = MagicMock(spec=HeavyIQConfig)
        config.rag_faiss_persist_dir = temp_persist_dir
        config.rag_embed_dimension = 128
        config.rag_vectordb_type = "faiss"
        return config

    def test_concurrent_add_operations(self, mock_config, temp_persist_dir):
        """Test that concurrent add operations don't corrupt the index."""
        import threading
        import heavyrag.vector_stores.faiss as faiss_module
        from heavyrag.vector_stores.faiss import FaissIQVectorStore
        from llama_index.core.schema import TextNode

        # Reset global state
        original_index = faiss_module.faiss_index
        faiss_module.faiss_index = None
        faiss_module.last_update_time = 0

        try:
            dimension = mock_config.rag_embed_dimension
            store = FaissIQVectorStore(
                persist_dir=temp_persist_dir,
                dimension=dimension,
            )

            errors = []
            added_ids = []
            lock = threading.Lock()

            def add_nodes(thread_id):
                try:
                    for i in range(5):
                        node_id = generate_hex_id()
                        node = TextNode(
                            text=f"Thread {thread_id} doc {i}",
                            id_=node_id,
                            embedding=np.random.rand(dimension).tolist(),
                            metadata={"thread": thread_id, "doc": i},
                        )
                        ids = store.add([node])
                        with lock:
                            added_ids.extend(ids)
                except Exception as e:
                    with lock:
                        errors.append(f"Thread {thread_id}: {e}")

            # Run multiple threads adding nodes concurrently
            threads = [threading.Thread(target=add_nodes, args=(i,)) for i in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0, f"Errors during concurrent adds: {errors}"
            # Should have 4 threads * 5 docs = 20 nodes
            assert len(added_ids) == 20, f"Expected 20 nodes, got {len(added_ids)}"

            # Verify index integrity
            index_ids = store.get_ids()
            assert len(index_ids) == 20, f"Index has {len(index_ids)} entries, expected 20"

        finally:
            faiss_module.faiss_index = original_index

    def test_concurrent_persist_operations(self, mock_config, temp_persist_dir):
        """Test that concurrent persist operations don't corrupt files."""
        import threading
        import heavyrag.vector_stores.faiss as faiss_module
        from heavyrag.vector_stores.faiss import FaissIQVectorStore
        from llama_index.core.schema import TextNode

        # Reset global state
        original_index = faiss_module.faiss_index
        faiss_module.faiss_index = None
        faiss_module.last_update_time = 0

        try:
            dimension = mock_config.rag_embed_dimension
            store = FaissIQVectorStore(
                persist_dir=temp_persist_dir,
                dimension=dimension,
            )

            # Add some initial nodes
            for i in range(10):
                node = TextNode(
                    text=f"Initial doc {i}",
                    id_=generate_hex_id(),
                    embedding=np.random.rand(dimension).tolist(),
                    metadata={"initial": True},
                )
                store.add([node])

            errors = []

            def persist_worker(worker_id, iterations):
                try:
                    for _ in range(iterations):
                        store.persist()
                except Exception as e:
                    errors.append(f"Worker {worker_id}: {e}")

            # Run multiple threads persisting concurrently
            threads = [
                threading.Thread(target=persist_worker, args=(i, 5))
                for i in range(4)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0, f"Errors during concurrent persists: {errors}"

            # Verify files are valid
            index_file = os.path.join(temp_persist_dir, "faiss_index.bin")
            assert os.path.exists(index_file)
            
            # Try to load the index to verify it's not corrupted
            loaded_index = faiss.read_index(index_file)
            assert loaded_index.ntotal == 10

        finally:
            faiss_module.faiss_index = original_index

    def test_atomic_persist_no_partial_writes(self, mock_config, temp_persist_dir):
        """Test that atomic persist doesn't leave partial files on failure."""
        import heavyrag.vector_stores.faiss as faiss_module
        from heavyrag.vector_stores.faiss import FaissIQVectorStore
        from llama_index.core.schema import TextNode

        # Reset global state
        original_index = faiss_module.faiss_index
        faiss_module.faiss_index = None

        try:
            dimension = mock_config.rag_embed_dimension
            store = FaissIQVectorStore(
                persist_dir=temp_persist_dir,
                dimension=dimension,
            )

            # Add a node and persist successfully first
            node = TextNode(
                text="Test doc",
                id_=generate_hex_id(),
                embedding=np.random.rand(dimension).tolist(),
                metadata={"test": True},
            )
            store.add([node])
            store.persist()

            # Verify the persist worked
            index_file = os.path.join(temp_persist_dir, "faiss_index.bin")
            assert os.path.exists(index_file)

            # No temp files should remain after successful persist
            temp_files = [f for f in os.listdir(temp_persist_dir) if f.endswith('.tmp')]
            assert len(temp_files) == 0, f"Temp files found: {temp_files}"

        finally:
            faiss_module.faiss_index = original_index


class TestFaissMultiprocessWrites:
    """Tests for FAISS writes across multiple processes (simulating gunicorn workers)."""

    @pytest.fixture
    def temp_persist_dir(self):
        """Create a temporary directory for FAISS persistence."""
        temp_dir = tempfile.mkdtemp(prefix="faiss_multiprocess_test_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    # @pytest.mark.skip(reason="Multiprocess FAISS tests can cause issues with FAISS global state. Run in isolation.")
    def test_multiprocess_concurrent_adds(self, temp_persist_dir):
        """Test that multiple processes can safely add to FAISS with file locking."""
        dimension = 128
        num_workers = 4
        
        # Use spawn to ensure clean process state
        ctx = multiprocessing.get_context('spawn')
        result_queue = ctx.Queue()
        
        processes = []
        for i in range(num_workers):
            p = ctx.Process(
                target=_multiprocess_worker,
                args=(temp_persist_dir, dimension, i, result_queue)
            )
            processes.append(p)
        
        # Start all processes
        for p in processes:
            p.start()
        
        # Wait for all to complete
        for p in processes:
            p.join(timeout=30)
        
        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        # Check for errors
        errors = [r for r in results if r["error"] is not None]
        assert len(errors) == 0, f"Process errors: {errors}"
        
        # Check all workers completed
        assert len(results) == num_workers, f"Expected {num_workers} results, got {len(results)}"
        
        # Verify total adds
        total_added = sum(r["added"] for r in results)
        assert total_added == num_workers * 5, f"Expected {num_workers * 5} adds, got {total_added}"

    def test_file_lock_works_across_threads(self, temp_persist_dir):
        """Test file locking mechanism with threads (easier to verify than processes)."""
        import threading
        from heavyrag.vector_stores.faiss import FaissFileLock
        
        lock_file = os.path.join(temp_persist_dir, ".faiss.lock")
        file_lock = FaissFileLock(lock_file, timeout=10.0)
        
        # Track execution order
        execution_order = []
        order_lock = threading.Lock()
        
        def worker(worker_id):
            with file_lock.write_lock():
                with order_lock:
                    execution_order.append(f"{worker_id}_start")
                # Simulate work
                time.sleep(0.1)
                with order_lock:
                    execution_order.append(f"{worker_id}_end")
        
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Verify no overlapping execution (each start should be followed by its end before next start)
        for i in range(0, len(execution_order) - 1, 2):
            start_event = execution_order[i]
            end_event = execution_order[i + 1]
            worker_id = start_event.split("_")[0]
            assert end_event == f"{worker_id}_end", f"Overlapping execution detected: {execution_order}"


class TestFaissController:
    """Tests for the FAISS controller."""

    @pytest.fixture
    def temp_persist_dir(self):
        """Create a temporary directory for FAISS persistence."""
        temp_dir = tempfile.mkdtemp(prefix="faiss_controller_test_")
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def mock_config(self, temp_persist_dir):
        """Create a mock config for testing."""
        config = MagicMock()
        config.rag_faiss_persist_dir = temp_persist_dir
        config.rag_embed_dimension = 384
        config.rag_vectordb_type = "faiss"
        config.rag_embed_server_base = "http://localhost:8080"
        return config

    def test_faiss_controller_persist_dir(self, mock_config):
        """Test FaissController persist_dir property."""
        with patch("heavyrag.controller.CONFIG", mock_config):
            from heavyrag.controller import FaissController

            controller = FaissController()

            assert controller.persist_dir == mock_config.rag_faiss_persist_dir

    def test_faiss_controller_get_vectorstore_caches(self, mock_config, temp_persist_dir):
        """Test FaissController caches the vectorstore."""
        import heavyrag.vector_stores.faiss as faiss_module

        # Reset global state so FaissIQVectorStore creates a fresh index
        faiss_module.faiss_index = None
        faiss_module.last_update_time = 0

        with patch("heavyrag.controller.CONFIG", mock_config), patch(
            "heavyrag.controller.EMBED_MODEL", MagicMock()
        ):
            from heavyrag.controller import FaissController

            controller = FaissController()
            controller._cached_vector_store = None

            # First call should create
            store1 = controller.get_vectorstore()

            # Second call should return cached
            store2 = controller.get_vectorstore()

            assert store1 is store2


