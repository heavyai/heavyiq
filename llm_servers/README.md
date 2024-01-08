## Custom LLM Servers

Folloing LLM servers can be built and run locally as docker containers with corresponding LLM models. 

1. Whisper-CPP
2. LLAMA-CPP (WIP)
3. VLLM (WIP)

### How to run Whisper-CPP server?


Build the image from Dockerfile and then run it. For now, `ggml-tiny` model has been taken and I'm sure that it would be enough for our usecases. 

```
docker-compose build whisper-cpp
docker-compose up whisper-cpp
```
