# Run FastAPI application with Uvicorn
run-fastapi:
	gunicorn 'heavyiq.api:create_app("./config.toml")' --config gunicorn_conf.py -w 2
