FROM python:3.10 AS obfuscator

WORKDIR /usr/src/app

COPY pyarmor-regfile-5130.zip ./
COPY requirements.txt ./
COPY requirements-dev.txt ./
COPY heavynl/ ./heavynl/

RUN pip install --no-cache-dir -r requirements-dev.txt
RUN pyarmor reg pyarmor-regfile-5130.zip
RUN pyarmor gen ./heavynl

FROM python:3.10 AS runner

WORKDIR /usr/src/app

COPY --from=obfuscator /usr/src/app/dist/ ./
COPY --from=obfuscator /usr/src/app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

EXPOSE 8000
# config_path needs to be given to get_app as a string (get_app(config_path="config.toml"))
CMD ["gunicorn", "-w", "4", "heavynl.api:get_app()"]
