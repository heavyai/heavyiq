FROM python:3.10

WORKDIR /usr/src/app

COPY config.toml ./
COPY requirements.txt ./
COPY gunicorn.conf.py ./
COPY heavyiq/ ./heavyiq/
COPY heavyrag/ ./heavyrag/

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

EXPOSE 8000
CMD ["gunicorn", "-b", ":8000", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--preload", "heavyiq.api:create_app()"]
