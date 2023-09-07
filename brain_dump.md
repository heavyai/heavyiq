# HeavyIQ Tech Dump

## Table of Contents

- [Onboarding](#onboarding)
  - [Notable Modules](#notable-modules)
  - [Gotchyas](#gotchyas)
- [Models](#models)
  - [Background](#background)
  - [Glossary](#glossary)
  - [Custom Models](#custom-models)
- [Chains](#chains)
- [Indexes](#indexes)
  - [HeavyDB](#heavydb-index)
  - [Documentation](#documentation-index)
- [Future Dev Musings](#future-dev-musings)

## Onboarding

Welcome to HeavyIQ! This document is intended to be a quick reference for developers new to the HeavyIQ codebase. It is not intended to be a comprehensive guide to the codebase, but rather a quick reference for common tasks, concepts and a couple quirks to keep in mind ('gotchyas').

### Notable Modules

#### **Langchain**

[Documentation](https://python.langchain.com/)

LangChain is a framework for developing applications powered by language models. It contains many useful components for building applications that use language models. The most important of these are:
* [Models](https://python.langchain.com/docs/modules/model_io/models/)
* [Prompts](https://python.langchain.com/docs/modules/model_io/prompts/)
* [Chains](https://python.langchain.com/docs/modules/chains/)
* [Vector Stores](https://python.langchain.com/docs/modules/data_connection/vectorstores/)

There is an important component in LangChain that is not listed above as it is not yet used in production, but may become important in future development. This is the [Agent](https://python.langchain.com/docs/modules/agents/). The core idea of agents is to use an LLM to choose a sequence of actions to take. In **chains**, a sequence of actions is hardcoded (in code). In **agents**, a language model is used as a reasoning engine to determine which actions to take and in which order. Agents are a very powerful concept, but there is a trade-off because agents call the LLM multiple times, resulting in a significant increase in latency. Mitigations strategy for this include streaming results from the LLM, caching results, and using a smaller customized LLM with lower latency.

There are more components in LangChain that are useful to be aware of as building blocks for future development. These include:
* [Memory](https://python.langchain.com/docs/modules/memory/)
* [Callbacks](https://python.langchain.com/docs/modules/callbacks/)

#### **ChromaDB**

[Documentation](https://docs.trychroma.com/)

ChromaDB is the vector store used by HeavyIQ. A vector store is a document database that stores [embeddings](https://python.langchain.com/docs/modules/data_connection/text_embedding/), or mathematical representations of the semantic meanings of text. This allows us to store and query documents based on their semantic meaning, rather than their literal text. Documents retrieved from the vector store are typically used to provide relevant context to a language model. As a reminder, language models only have a general understanding of the materials they are trained on, so vector stores are used to provide information to the model that it was not trained on. For example, the contents of the HeavyDB documentation or the contents of a specific HeavyDB database.

**High-level Overview**
1. Documents to be indexed are loaded using a [document loader](https://python.langchain.com/docs/modules/data_connection/document_loaders/)
2. The documents are split using a [text splitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter). The purpose of splitting the text is dual purpose:
    a. The embedding created for each 'sub-document' is closer to the semantic meaning of the text than the embedding of the entire document.
    b. The chunks of text retrieved from the vector store are smaller, preventing the prompt from becoming too long as to exceed the model's "context limit" (the maximum amount of tokens a model can process at once).
3. Embeddings are created for each sub-document using a [text embedding](https://python.langchain.com/docs/modules/data_connection/text_embedding/) and inserted into the vector store along with the document's metadata. The metadata is used to identify the document when retrieving it from the vector store, such as the URL of the document or the name of the database it belongs to.
4. Querying the vector store is done using a [vector store retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/). The query text is converted into a text embedding and the vector store is queried for the most similar embeddings. The metadata of the most similar embeddings is returned with a list of documents.
5. These documents are typically used in a [document chain](https://python.langchain.com/docs/modules/chains/document/) to summarize, answer questions, or extract information from the documents.

#### **Pydantic**

[Documentation](https://docs.pydantic.dev/1.10/)

Pydantic is a very useful data validation module that uses Python's class inheritence to describe/enforce class variables and types. Pydantic is a staple in modern python modules and langchain and fastapi both use it extensively. All classes that inherit Pydantic's `BaseModel` enforce data types at runtime and they also gain the ability to easily be serialized to and from JSON using the `dict()` method.

#### **FastAPI**

[Documentation](https://fastapi.tiangolo.com/)

FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints (via Pydantic). FastAPI is used to build the HeavyIQ API and is the primary interface for interacting with HeavyIQ. FastAPI automatically generates [Swagger Docs](https://swagger.io/docs/) which is an easy way to interact with the API and test specific endpoints in a dev or production environment. The Swagger Docs are available at the `/docs` endpoint of the API.

**Note:** FastAPI makes use of asynchronous endpoint handlers for better performance. Async code is python is similar to that of javascript, using `async` to define async functions and using `await` to call async functions.

#### **ConfZ**

[Documentation](https://confz.readthedocs.io/en/latest/)

ConfZ is a config-file parsing module that is used to load configuration files for HeavyIQ. The config-file is converted into a python class that can be accessed as a singleton. The config-file is loaded at startup and used throughout the codebase. ConfZ uses Pydantic to enforce data types, but sometimes more thorough validation is required (i.e. checking that an API key is valid, or that certain optional configuration values are set in particular cases). All configuration options for HeavyIQ can be found in the `heavyiq/config/config_schema.py` file.

### Gotchyas

**Config Race Conditions**

HeavyIQ can be supplied a path to the config file at startup which is optimal for our various deployment methods. However, this means that the config file is not available at import time and the config file can only be retrieved after the initial call is made with the config's path. Prior to this change, `get_config` could be called at import time, but this is no longer the case. In most cases, moving the `get_config` call into the methods the config is required in will solve this issue, but in the case of the loggers (`heavyiq/logging_utils.py`) the solution was a little more convoluted. Instead of being able to import the loggers directly, the loggers are now initialized manually after the config is loaded and must be retrieved at runtime using the various 'get_logger' methods. This is a bit of a pain, but it is the only way to ensure that the config is loaded before the loggers are initialized. Another example of this can be found in `heavyiq/langchain/utils.py`, where we must manually init the telemetrics configuration after the config is loaded.

**Custom LLM Types**

Although OpenAI's GPT models are generally equipped for a wide variety of tasks, local models are often trained on a specific task and are not equipped for other tasks. For example, a model that was trained to translate natural language queries into SQL queries might be unable to use SQL query results to answer a user's question. The code found in `heavyiq/langchain/llms/__init__.py` provides the means with which to init and retrieve different models for different purposes by means of the `LLMType` enum. When the config specifies that custom models are to be used instead of OpenAI (or Azure OpenAI), you can correlate that model to a specific `LLMType` and use that `LLMType` to retrieve the model. The default `LLMType.ANY` is used to retrieve the default model, which in the case of OpenAI will be the OpenAI model.

**Response Parsing**

## Models

### Background

There are two main types of LLMs, text completion and conversational (chat). A text completion model is used to complete a prompt with a single response. A conversational model is trained to use a list of messages as context and then generate a response. The response is then added to the context and the process is repeated, allowing the LLM to keep a working memory of the conversation. When HeavyIQ is configured to use OpenAI's API, the best performing models are typically conversational models, however, Heavy's current efforts in training custom models are focused on text completion models. A text completion model can somewhat emulate the behavior of a conversational model if the message history is provided in the prompt, but this is not ideal. Luckily, LangChain's LLM integration is flexible enough to allow for the use of both types of models, but it is important to understand the differences between the two as any future features that require maintaining the context of an interaction between calls to the LLM will benefit from the use of a conversational model.

### Glossary
* **LLM** - Large Language Model
* **Token** - A unit of text that a language model reads when processing input. In English, a token could be as short as one character or as long as one word (e.g., 'a', 'apple'). The total number of tokens in a model's input affects the model's speed and ability to generate responses.
* **Prompt** - The text that is provided to the LLM as input. The prompt is typically a question or a statement that the LLM will complete. A prompt can also include formatting instructions to inform the structure of the response.
* **Response** - The text that is generated by the LLM in response to the prompt.
* **Context** - The text that is provided to the LLM as input. The context is typically a list of messages that the LLM will use to generate a response.
* **Temperature** - Temperature is a parameter that controls the randomness of the LLM's responses. A higher temperature will result in more random responses, while a lower temperature will result in more predictable responses. A temperature of 0.0 is preferable for tasks that require a precise predictable response, such as natural language to SQL translation, but higher temperatures may be preferably for tasks that require more creativity, such as summarization or question answering.
* **Beam Search** - Beam search is an algorithm that generates multiple responses in parallel until it finds an optimal string of tokens. The actual implementation of this algorithm take place on the LLM server, but the number of beams can be configured in the config for LLM servers that support it (VLLM as of this writing).

### Custom Models

There are many different custom models available, but there are limited solutions for hosting them. For ease of integration with LangChain, we have focused our efforts on servers that provide a standardized REST interface to the underlying model. HeavyIQ currently supports two different custom llm servers, [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) and [vLLM](https://github.com/vllm-project/vllm). vLLM currently supports more models and includes niceties such as automatic huggingface model downloads and beam search. See Todd for more information regarding the pros/cons of each choice. I predict we will use vLLM moving forward, but it's important to note the codebase accomodates both. Most of the code pertaining to custom models is in a centralized location (`heavyiq/langchain/llms`), so it should be fairly easy to reason about.

## Chains

### HeavyDB

### HeavyDB Metadata

### Documentation

## Indexes

### HeavyDB Metadata Index

### HeavyAI Documentation Index

## Future Dev Musings
