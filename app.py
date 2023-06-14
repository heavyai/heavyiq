from typing import Any
from flask import request
from heavynl.api import get_app
from heavynl.logging_utils import app_logger, heavynl_logger

app = get_app()


@app.after_request
def log_request(response: Any) -> Any:
    request.response = response
    app_logger.info("Request:", extra={"response": response})
    return response


@app.before_request
def log_request_body():
    # Log the request body
    heavynl_logger.debug("Request Body: %s", request.get_data(as_text=True))
