FROM tiangolo/uwsgi-nginx-flask:python3.10

COPY ./requirements.txt /app/requirements.txt
COPY ./modules /app/modules
COPY ./app.py /app/main.py

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
