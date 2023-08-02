HeavyIQ is not limited to using OpenAI for its large language model (LLM). Below are instructions on how to self-host an OpenAI API-compliant server that HeavyIQ can communicate with.

## Installation

Requires python 3.8.1 or above

1. Create a new virtual environment

```bash
$ python3 -m venv venv
$ . venv/bin/activate
```

2. Install
a. **For CPU**
    ```bash
    $ pip install -r requirements.txt
    ```
    b. **For GPU**
    To install with OpenBLAS, set the `LLAMA_OPENBLAS=1` environment variable before installing:

    ```bash
    CMAKE_ARGS="-DLLAMA_OPENBLAS=on" FORCE_CMAKE=1 pip install -r requirements.txt
    ```

    To install with cuBLAS, set the `LLAMA_CUBLAS=1` environment variable before installing:

    ```bash
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install -r requirements.txt
    ```

    To install with CLBlast, set the `LLAMA_CLBLAST=1` environment variable before installing:

    ```bash
    CMAKE_ARGS="-DLLAMA_CLBLAST=on" FORCE_CMAKE=1 pip install -r requirements.txt
    ```

    To install with Metal (MPS), set the `LLAMA_METAL=on` environment variable before installing:

    ```bash
    CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install -r requirements.txt
    ```

    Detailed MacOS documentation [available here](https://github.com/abetlen/llama-cpp-python/blob/main/docs/macos_install.md)

3. Download a model
For example, [this one](https://huggingface.co/TheBloke/open-llama-7b-open-instruct-GGML/blob/main/open-llama-7B-open-instruct.ggmlv3.q4_0.bin)

## Running Locally

1. With venv active (`. venv/bin/activate`)
    ```bash
    $ python -m llama_cpp.server --host 0.0.0.0 --model PATH/TO/MODEL.bin
2. Visit `http://localhost:8000/docs` for API Documentation

## Running in Docker

1.
```bash
docker run --rm -it -p 8000:8000 \
-v /path/to/models:/models \
-e MODEL=/models/MODEL_NAME.bin \
ghcr.io/abetlen/llama-cpp-python:latest
```

## Configuring HeavyIQ to use this server

**TBD**
