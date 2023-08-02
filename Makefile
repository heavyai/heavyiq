# Run Flask application
run-flask:
	flask run

# Run FastAPI application with Uvicorn
run-fastapi:
	uvicorn fast_app:app --reload
