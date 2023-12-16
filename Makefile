# Run FastAPI application with Uvicorn
run-fastapi:
	uvicorn app:app --reload

run-test:
	pytest tests/api tests/config tests/langchain  --disable-warnings -rs --config-path=./config.test.toml
