# Run HeavyIQ application with Uvicorn
run-fastapi:
	uvicorn app:app --reload
# Run HeavyIQ application with gunicorn
run-gunicorn:
	gunicorn --preload --capture-output -t 0 -w 4 -k uvicorn.workers.UvicornWorker -c 'gunicorn.conf.py' 'heavyiq.api:create_app("./config.toml")'
#Run HeavyIQ tests
run-test:
	pytest tests --disable-warnings -rs --config-path="./config.test.toml"
