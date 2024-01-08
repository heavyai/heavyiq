#!/bin/bash

# Check if the model file exists
if [ ! -f "/app/models/ggml-tiny.bin" ]; then
    # Download the model file if it doesn't exist
    wget -O "/app/models/ggml-tiny.bin" "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin?download=true"
fi

# Start the Whisper CPP server with the specified model
exec python -m whisper_cpp_python.server --model /app/models/ggml-tiny.bin
