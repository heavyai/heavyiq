"""
Tests for the shared_state module.

Tests both standalone mode (default) and manager mode (multiprocessing).
"""

import multiprocessing
import threading
import time
from unittest.mock import patch

import pytest

from heavyiq import shared_state


class TestSharedStateStandalone:
    """Tests for standalone mode (no multiprocessing manager)."""

    @pytest.fixture(autouse=True)
    def reset_shared_state(self):
        """Reset shared state before each test."""
        # Save original state
        original_dict = shared_state._shared_dict
        original_lock = shared_state._shared_lock
        original_manager = shared_state._manager
        original_standalone = shared_state._is_standalone
        original_address = shared_state._manager_address
        original_authkey = shared_state._manager_authkey

        # Reset to clean state
        shared_state._shared_dict = None
        shared_state._shared_lock = None
        shared_state._manager = None
        shared_state._is_standalone = True
        shared_state._manager_address = None
        shared_state._manager_authkey = None

        yield

        # Restore original state
        shared_state._shared_dict = original_dict
        shared_state._shared_lock = original_lock
        shared_state._manager = original_manager
        shared_state._is_standalone = original_standalone
        shared_state._manager_address = original_address
        shared_state._manager_authkey = original_authkey

    def test_init_standalone(self):
        """Test standalone initialization."""
        shared_state.init_standalone()

        assert shared_state._shared_dict is not None
        assert shared_state._shared_lock is not None
        assert shared_state._is_standalone is True
        assert isinstance(shared_state._shared_dict, dict)
        assert isinstance(shared_state._shared_lock, type(threading.Lock()))

    def test_auto_init_on_get(self):
        """Test that get() auto-initializes standalone mode."""
        assert shared_state._shared_dict is None

        # Getting should auto-initialize
        result = shared_state.get("nonexistent_key")

        assert result is None
        assert shared_state._shared_dict is not None
        assert shared_state._is_standalone is True

    def test_put_and_get(self):
        """Test basic put and get operations."""
        shared_state.init_standalone()

        shared_state.put("test_key", "test_value")
        result = shared_state.get("test_key")

        assert result == "test_value"

    def test_put_and_get_various_types(self):
        """Test put/get with various data types."""
        shared_state.init_standalone()

        # String
        shared_state.put("string_key", "string_value")
        assert shared_state.get("string_key") == "string_value"

        # Integer
        shared_state.put("int_key", 42)
        assert shared_state.get("int_key") == 42

        # Float
        shared_state.put("float_key", 3.14)
        assert shared_state.get("float_key") == 3.14

        # List
        shared_state.put("list_key", [1, 2, 3])
        assert shared_state.get("list_key") == [1, 2, 3]

        # Dict
        shared_state.put("dict_key", {"nested": "value"})
        assert shared_state.get("dict_key") == {"nested": "value"}

        # None
        shared_state.put("none_key", None)
        assert shared_state.get("none_key") is None

    def test_delete(self):
        """Test delete operation."""
        shared_state.init_standalone()

        shared_state.put("to_delete", "value")
        assert shared_state.get("to_delete") == "value"

        shared_state.delete("to_delete")
        assert shared_state.get("to_delete") is None

    def test_delete_nonexistent_key(self):
        """Test deleting a non-existent key doesn't raise."""
        shared_state.init_standalone()

        # Should not raise
        shared_state.delete("nonexistent_key")

    def test_get_nonexistent_key(self):
        """Test getting a non-existent key returns None."""
        shared_state.init_standalone()

        result = shared_state.get("nonexistent_key")
        assert result is None

    def test_overwrite_value(self):
        """Test overwriting an existing value."""
        shared_state.init_standalone()

        shared_state.put("key", "value1")
        assert shared_state.get("key") == "value1"

        shared_state.put("key", "value2")
        assert shared_state.get("key") == "value2"

    def test_lock_context_manager(self):
        """Test lock context manager."""
        shared_state.init_standalone()

        with shared_state.lock():
            shared_state.put("locked_key", "locked_value")
            value = shared_state.get("locked_key")

        assert value == "locked_value"

    def test_lock_auto_init(self):
        """Test that lock() auto-initializes standalone mode."""
        assert shared_state._shared_lock is None

        with shared_state.lock():
            pass

        assert shared_state._shared_lock is not None

    def test_create_dict_standalone(self):
        """Test create_dict returns regular dict in standalone mode."""
        shared_state.init_standalone()

        new_dict = shared_state.create_dict()

        assert isinstance(new_dict, dict)
        assert new_dict == {}

    def test_create_list_standalone(self):
        """Test create_list returns regular list in standalone mode."""
        shared_state.init_standalone()

        new_list = shared_state.create_list()

        assert isinstance(new_list, list)
        assert new_list == []

    def test_create_lock_standalone(self):
        """Test create_lock returns threading.Lock in standalone mode."""
        shared_state.init_standalone()

        new_lock = shared_state.create_lock()

        assert isinstance(new_lock, type(threading.Lock()))

    def test_is_standalone(self):
        """Test is_standalone function."""
        shared_state.init_standalone()

        assert shared_state.is_standalone() is True

    def test_get_manager_info_standalone(self):
        """Test get_manager_info in standalone mode."""
        shared_state.init_standalone()
        shared_state.put("key1", "value1")
        shared_state.put("key2", "value2")

        info = shared_state.get_manager_info()

        assert info["is_standalone"] is True
        assert info["manager_address"] is None
        assert info["manager_running"] is False
        assert info["dict_size"] == 2

    def test_get_shared_dict(self):
        """Test get_shared_dict function."""
        shared_state.init_standalone()

        d = shared_state.get_shared_dict()

        assert d is shared_state._shared_dict

    def test_shared_state_keys_enum(self):
        """Test SharedStateKeys enum values."""
        assert shared_state.SharedStateKeys.HeavyDBLicenseEdition.value == "heavydb_license_edition"
        assert shared_state.SharedStateKeys.ConfFilePath.value == "config_file_path"

    def test_concurrent_access_standalone(self):
        """Test concurrent access in standalone mode with threading."""
        shared_state.init_standalone()
        shared_state.put("counter", 0)

        def increment():
            for _ in range(100):
                with shared_state.lock():
                    count = shared_state.get("counter") or 0
                    shared_state.put("counter", count + 1)

        threads = [threading.Thread(target=increment) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert shared_state.get("counter") == 500


class TestSharedStateManager:
    """Tests for manager mode (multiprocessing)."""

    @pytest.fixture(autouse=True)
    def reset_shared_state(self):
        """Reset shared state before and after each test."""
        # Save original state
        original_dict = shared_state._shared_dict
        original_lock = shared_state._shared_lock
        original_manager = shared_state._manager
        original_standalone = shared_state._is_standalone
        original_address = shared_state._manager_address
        original_authkey = shared_state._manager_authkey

        # Reset to clean state
        shared_state._shared_dict = None
        shared_state._shared_lock = None
        shared_state._manager = None
        shared_state._is_standalone = True
        shared_state._manager_address = None
        shared_state._manager_authkey = None

        yield

        # Shutdown manager if started
        if shared_state._manager is not None:
            try:
                shared_state.shutdown_manager()
            except Exception:
                pass

        # Restore original state
        shared_state._shared_dict = original_dict
        shared_state._shared_lock = original_lock
        shared_state._manager = original_manager
        shared_state._is_standalone = original_standalone
        shared_state._manager_address = original_address
        shared_state._manager_authkey = original_authkey

    def test_start_manager(self):
        """Test starting the shared state manager."""
        shared_state.start_manager()

        assert shared_state._manager is not None
        assert shared_state._shared_dict is not None
        assert shared_state._shared_lock is not None
        assert shared_state._is_standalone is False
        assert shared_state._manager_address is not None
        assert shared_state._manager_authkey is not None

    def test_start_manager_idempotent(self):
        """Test that starting manager multiple times is safe."""
        shared_state.start_manager()
        manager1 = shared_state._manager
        address1 = shared_state._manager_address

        shared_state.start_manager()  # Should be no-op
        manager2 = shared_state._manager
        address2 = shared_state._manager_address

        assert manager1 is manager2
        assert address1 == address2

    def test_shutdown_manager(self):
        """Test shutting down the manager."""
        shared_state.start_manager()
        assert shared_state._manager is not None

        shared_state.shutdown_manager()

        assert shared_state._manager is None
        assert shared_state._shared_dict is None
        assert shared_state._shared_lock is None

    def test_shutdown_manager_idempotent(self):
        """Test that shutting down manager multiple times is safe."""
        shared_state.start_manager()
        shared_state.shutdown_manager()

        # Should not raise
        shared_state.shutdown_manager()

    def test_put_get_with_manager(self):
        """Test put/get with manager mode."""
        shared_state.start_manager()

        shared_state.put("manager_key", "manager_value")
        result = shared_state.get("manager_key")

        assert result == "manager_value"

    def test_delete_with_manager(self):
        """Test delete with manager mode."""
        shared_state.start_manager()

        shared_state.put("to_delete", "value")
        shared_state.delete("to_delete")

        assert shared_state.get("to_delete") is None

    def test_lock_with_manager(self):
        """Test lock with manager mode."""
        shared_state.start_manager()

        with shared_state.lock():
            shared_state.put("locked", "value")

        assert shared_state.get("locked") == "value"

    def test_create_dict_with_manager(self):
        """Test create_dict with manager mode."""
        shared_state.start_manager()

        new_dict = shared_state.create_dict()

        # Should be a manager dict proxy, not a regular dict
        assert new_dict is not None
        new_dict["key"] = "value"
        assert new_dict["key"] == "value"

    def test_create_list_with_manager(self):
        """Test create_list with manager mode."""
        shared_state.start_manager()

        new_list = shared_state.create_list()

        # Should be a manager list proxy
        assert new_list is not None
        new_list.append("item")
        assert "item" in new_list

    def test_is_standalone_with_manager(self):
        """Test is_standalone returns False with manager."""
        shared_state.start_manager()

        assert shared_state.is_standalone() is False

    def test_get_manager_info_with_manager(self):
        """Test get_manager_info with manager mode."""
        shared_state.start_manager()
        shared_state.put("key", "value")

        info = shared_state.get_manager_info()

        assert info["is_standalone"] is False
        assert info["manager_address"] is not None
        assert info["manager_running"] is True
        assert info["dict_size"] == 1

    def test_get_manager(self):
        """Test get_manager function."""
        shared_state.start_manager()

        manager = shared_state.get_manager()

        assert manager is shared_state._manager

    def test_connect_to_manager(self):
        """Test connect_to_manager function."""
        shared_state.start_manager()
        shared_state.put("before_connect", "value")

        # This should work and print a message
        shared_state.connect_to_manager()

        # Should still be able to access data
        assert shared_state.get("before_connect") == "value"

    def test_connect_to_manager_no_dict(self):
        """Test connect_to_manager when no dict available."""
        # Should not raise, just print a message
        shared_state.connect_to_manager()

    def test_concurrent_access_with_manager_threads(self):
        """Test concurrent access with manager mode using threads."""
        shared_state.start_manager()
        shared_state.put("counter", 0)

        def increment():
            for _ in range(100):
                with shared_state.lock():
                    count = shared_state.get("counter") or 0
                    shared_state.put("counter", count + 1)

        threads = [threading.Thread(target=increment) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 500 increments should be counted
        assert shared_state.get("counter") == 500

    def test_concurrent_access_with_manager_multiprocess(self):
        """
        Test concurrent access with manager mode using multiprocessing.Process.
        
        This demonstrates the real use case: multiple processes sharing state
        through the SyncManager, similar to gunicorn workers.
        
        Note: This test is skipped by default because FAISS + multiprocessing.Process
        causes segmentation faults. Run this test in isolation if needed.
        """
        import multiprocessing
        
        shared_state.start_manager()
        shared_state.put("process_counter", 0)
        
        shared_dict = shared_state._shared_dict
        shared_lock = shared_state._shared_lock
        
        def worker_process(worker_id, shared_dict_proxy, shared_lock_proxy):
            import time
            for i in range(10):
                shared_lock_proxy.acquire()
                try:
                    count = shared_dict_proxy.get("process_counter", 0)
                    time.sleep(0.001)
                    shared_dict_proxy["process_counter"] = count + 1
                finally:
                    shared_lock_proxy.release()

        processes = []
        for i in range(5):
            p = multiprocessing.Process(
                target=worker_process,
                args=(i, shared_dict, shared_lock)
            )
            processes.append(p)
            p.start()
        
        for p in processes:
            p.join()
            
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1)

        # All 50 increments (5 processes * 10 each) should be counted
        assert shared_state.get("process_counter") == 50

    def test_lock_prevents_race_condition(self):
        """
        Test that lock() prevents race conditions in read-modify-write operations.
        
        Without locking, concurrent increments would lose updates.
        With locking, all updates are preserved.
        """
        shared_state.start_manager()
        
        # Test WITHOUT lock (would fail with race condition)
        # We skip this as it's non-deterministic
        
        # Test WITH lock
        shared_state.put("safe_counter", 0)
        
        def safe_increment():
            for _ in range(50):
                with shared_state.lock():
                    # Read-modify-write is atomic within the lock
                    val = shared_state.get("safe_counter")
                    # Simulate some processing time
                    import time
                    time.sleep(0.001)
                    shared_state.put("safe_counter", val + 1)

        threads = [threading.Thread(target=safe_increment) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 200 increments preserved (4 threads * 50 each)
        assert shared_state.get("safe_counter") == 200

    def test_create_multiple_shared_dicts(self):
        """Test creating multiple independent shared dicts."""
        shared_state.start_manager()
        
        dict1 = shared_state.create_dict()
        dict2 = shared_state.create_dict()
        
        # They should be independent
        dict1["key"] = "value1"
        dict2["key"] = "value2"
        
        assert dict1["key"] == "value1"
        assert dict2["key"] == "value2"
        assert dict1 is not dict2


