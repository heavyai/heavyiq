"""
Shared state management for HeavyIQ.

This module provides a clean abstraction for shared state across Gunicorn workers.
The Manager process is started by Gunicorn's master process (on_starting hook),
and workers connect to it after fork (post_fork hook).

When running standalone (uvicorn, development), simple Python containers are used instead.

Usage:
    # For simple key-value state (individual ops are atomic)
    shared_state.put("key", "value")
    value = shared_state.get("key")
    
    # For compound operations that need atomicity across processes
    with shared_state.lock():
        count = shared_state.get("counter") or 0
        shared_state.put("counter", count + 1)
    
    # For creating shared caches (LRU, etc.)
    cache_dict = shared_state.create_dict()  # Returns manager.dict() or regular dict
    cache_list = shared_state.create_list()  # Returns manager.list() or regular list

Concurrency guarantees:
    - Individual get/put/delete operations are atomic (serialized by manager server)
    - Compound read-modify-write operations need explicit locking via lock()
    - In standalone mode, threading.Lock is used (sufficient for single-process)
"""

import os
import threading
from contextlib import contextmanager
from enum import Enum
from multiprocessing.managers import SyncManager
from typing import Any, Iterator

# Module-level state
_manager: SyncManager | None = None
_shared_dict: dict[Any, Any] | None = None
_shared_lock: Any = None  # manager.Lock() or threading.Lock()
_manager_address: tuple | None = None
_manager_authkey: bytes | None = None
_is_standalone: bool = True  # Default to standalone mode
_init_lock = threading.Lock()  # For initialization only (per-process is fine)


class SharedStateKeys(Enum):
    """Keys for well-known shared state entries."""
    HeavyDBLicenseEdition = "heavydb_license_edition"
    ConfFilePath = "config_file_path"


def start_manager() -> None:
    """
    Start the shared state manager process.
    
    This should ONLY be called from Gunicorn's on_starting hook (master process).
    Workers should call connect_to_manager() instead.
    """
    global _manager, _shared_dict, _shared_lock, _manager_address, _manager_authkey, _is_standalone
    
    with _init_lock:
        if _manager is not None:
            return  # Already started
        
        authkey = os.urandom(16)
        # Use standard SyncManager - it has dict/list/Lock already registered
        manager = SyncManager(address=("127.0.0.1", 0), authkey=authkey)
        manager.start()
        
        _manager = manager
        _shared_dict = manager.dict()  # Use manager.dict() directly
        _shared_lock = manager.Lock()  # Cross-process lock
        _manager_address = manager.address
        _manager_authkey = authkey
        _is_standalone = False
        
        print(f"[SharedState] Manager started at {_manager_address}, PID: {manager._process.pid}")


def connect_to_manager() -> None:
    """
    Verify connection to the shared state manager after fork.
    
    This should be called from Gunicorn's post_fork hook in worker processes.
    
    With --preload, workers inherit the proxy objects from the master process.
    The proxies maintain connection to the manager server and should work directly.
    This function verifies the connection is working.
    """
    if _shared_dict is None:
        print("[SharedState] No shared dict available (standalone mode or not initialized)")
        return
    
    try:
        # Verify the inherited proxy still works by accessing it
        _ = len(_shared_dict)
        print(f"[SharedState] Worker {os.getpid()} verified connection to manager at {_manager_address}")
    except Exception as e:
        print(f"[SharedState] Worker {os.getpid()} failed to access shared dict: {e}")
        print("[SharedState] Worker will continue but shared state may not work correctly")


def shutdown_manager() -> None:
    """
    Shutdown the shared state manager.
    
    This should be called from Gunicorn's on_exit hook.
    """
    global _manager, _shared_dict, _shared_lock, _manager_address, _manager_authkey
    
    with _init_lock:
        if _manager is not None:
            try:
                _manager.shutdown()
                print("[SharedState] Manager shut down cleanly")
            except Exception as e:
                print(f"[SharedState] Error shutting down manager: {e}")
            finally:
                _manager = None
                _shared_dict = None
                _shared_lock = None
                _manager_address = None
                _manager_authkey = None


def init_standalone() -> None:
    """
    Initialize standalone mode (no multiprocessing).
    
    This should be called when running without Gunicorn (e.g., uvicorn directly).
    Uses a simple dict instead of a shared manager dict.
    """
    global _shared_dict, _shared_lock, _is_standalone
    
    with _init_lock:
        if _shared_dict is None:
            _shared_dict = {}
            _shared_lock = threading.Lock()  # Per-process lock (sufficient for standalone)
            _is_standalone = True
            print("[SharedState] Initialized in standalone mode (local dict)")


def is_standalone() -> bool:
    """Return True if running in standalone mode (no shared manager)."""
    return _is_standalone


def get_shared_dict() -> dict[Any, Any]:
    """
    Get the shared dict (or local dict in standalone mode).
    
    Returns:
        The shared/local dict for storing cross-worker state.
        
    Raises:
        RuntimeError: If neither manager nor standalone mode is initialized.
    """
    if _shared_dict is None:
        # Auto-initialize standalone mode if nothing is set up
        init_standalone()
    return _shared_dict  # type: ignore


def get(key: Any) -> Any:
    """Get a value from shared state."""
    return get_shared_dict().get(key)


def put(key: Any, value: Any) -> None:
    """Put a value into shared state."""
    get_shared_dict()[key] = value


def delete(key: Any) -> None:
    """Delete a key from shared state."""
    try:
        del get_shared_dict()[key]
    except KeyError:
        pass


def get_manager_info() -> dict[str, Any]:
    """Get information about the manager state (for debugging)."""
    return {
        "is_standalone": _is_standalone,
        "manager_address": _manager_address,
        "manager_running": _manager is not None and _manager._process.is_alive() if _manager else False,
        "dict_size": len(_shared_dict) if _shared_dict else 0,
    }


# =============================================================================
# Cross-process locking
# =============================================================================

@contextmanager
def lock() -> Iterator[None]:
    """
    Acquire the global shared lock for compound operations.
    
    Use this when you need to perform read-modify-write operations atomically
    across all worker processes.
    
    Example:
        with shared_state.lock():
            count = shared_state.get("counter") or 0
            shared_state.put("counter", count + 1)
    
    In standalone mode, uses a threading.Lock (sufficient for single-process).
    In Gunicorn mode, uses manager.Lock() (cross-process synchronization).
    """
    global _shared_lock
    
    if _shared_lock is None:
        # Auto-initialize if needed
        init_standalone()
    
    _shared_lock.acquire()
    try:
        yield
    finally:
        _shared_lock.release()


def create_lock() -> Any:
    """
    Create a new independent cross-process lock.
    
    Use this when you need a dedicated lock for a specific resource,
    separate from the global shared lock.
    
    Returns:
        A manager.Lock() in Gunicorn mode, or threading.Lock() in standalone mode.
    """
    if _is_standalone or _manager is None:
        return threading.Lock()
    return _manager.Lock()


# =============================================================================
# Factory methods for creating shared containers (for caches, etc.)
# =============================================================================

def create_dict() -> dict[Any, Any]:
    """
    Create a new shared dict proxy (or regular dict in standalone mode).
    
    Use this for creating cache storage that needs to be shared across workers.
    Each call creates a NEW independent dict.
    
    Returns:
        A manager.dict() proxy in Gunicorn mode, or a regular dict in standalone mode.
    """
    if _is_standalone or _manager is None:
        return {}
    return _manager.dict()


def create_list() -> list[Any]:
    """
    Create a new shared list proxy (or regular list in standalone mode).
    
    Use this for creating ordered storage that needs to be shared across workers.
    Each call creates a NEW independent list.
    
    Returns:
        A manager.list() proxy in Gunicorn mode, or a regular list in standalone mode.
    """
    if _is_standalone or _manager is None:
        return []
    return _manager.list()


def get_manager() -> SyncManager | None:
    """
    Get the underlying manager instance.
    
    Returns None in standalone mode. Use create_dict()/create_list() instead
    for most use cases.
    """
    return _manager

