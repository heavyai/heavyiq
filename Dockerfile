FROM python:3.10 AS obfuscator

WORKDIR /usr/src/app

COPY config.toml ./
COPY pyarmor-regfile-5130.zip ./
COPY requirements.txt ./
COPY requirements-dev.txt ./
COPY heavynl/ ./heavynl/

RUN pip install --no-cache-dir pyarmor
RUN pyarmor reg pyarmor-regfile-5130.zip
RUN pyarmor gen ./heavynl

FROM python:3.10 AS runner

WORKDIR /usr/src/app

COPY --from=obfuscator /usr/src/app/dist/ ./
COPY --from=obfuscator /usr/src/app/requirements.txt ./requirements.txt
COPY --from=obfuscator /usr/src/app/config.toml ./config.toml
COPY --from=obfuscator /usr/src/app/heavynl/langchain/llama_model/ ./heavynl/langchain/llama_model/
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

EXPOSE 8000
CMD ["gunicorn", "-b", ":8000", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--preload", "heavynl.fastapi:create_app()"]
