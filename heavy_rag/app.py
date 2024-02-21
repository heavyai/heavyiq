import logging
import os

import uvicorn
from fastapi import FastAPI
from llama_index.core import Settings as IndexSettings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from rag.router import router
from settings import Settings

os.environ["ANONYMIZED_TELEMETRY"] = "false"


def create_app(settings: Settings) -> FastAPI:
    # Create FastAPI app instance
    app = FastAPI(title=settings.app_name)
    # set the embed model globally
    IndexSettings.embed_model = HuggingFaceEmbedding(model_name=settings.hf_embedding_model)

    # Configure logging
    log_level = logging.DEBUG if settings.debug else logging.INFO
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=log_level, format=log_format)
    logging.info("Settings: \n{}".format(settings.model_dump_json(indent=4)))

    # Include routers
    app.include_router(router, prefix="/api")

    return app


if __name__ == "__main__":
    # Initialize your settings
    settings = Settings()
    # Create FastAPI app
    app = create_app(settings)
    # Run the FastAPI app using Uvicorn server
    uvicorn.run(app, host=settings.host, port=settings.port)
