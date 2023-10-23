import threading
from typing import Optional


class RequestStatusManager:
    """
    A singleton used to track the statuses of requests that hit the API. As requests are processed this
    will be updated, allowing the API to send back to the user the current status of their request. 
    """

    _instance: Optional["RequestStatusManager"] = None
    _lock: threading.Lock = threading.Lock()
    request_status: dict[str, str] = {} 

    def __new__(cls) -> "RequestStatusManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(RequestStatusManager, cls).__new__(cls)
        return cls._instance

    def update_status(self, session_id, status) -> None:
        with self._lock:
            self.request_status[session_id] = status

    def get_status(self, session_id) -> str | None:
        with self._lock:
            return self.request_status.get(session_id)