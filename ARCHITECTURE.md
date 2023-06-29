# Architecture

## Introduction

HeavyIQ is a Python module designed to provide a natural language interface to [HeavyDB](https://github.com/heavyai/heavydb) using large language models and embeddings databases (also known as vector-stores). The goal of HeavyIQ is to use natural language processing to enable users of HeavyDB to interactively communicate with and gain insight into their data using their native language (as opposed to writing SQL or Python).

**HeavyIQ is made available via three interfaces:**
* [OpenAPI 3.0 Specification](https://github.com/OAI/OpenAPI-Specification/blob/main/versions/3.0.3.md) Compliant REST API
* Command-line Interface
* Python module


## Background and Context

A large language model (or LLM) is an artificial intelligence system that is trained on vast amounts of text data to understand the nuances of human language. It uses this knowledge to generate human-like responses to natural language queries and tasks (a.k.a. ‘prompts’), such as answering questions or completing sentences.

LLMs, such as GPT-4, lack knowledge beyond the data they were trained on, hence they do not possess any inherent knowledge of the contents of the HeavyDB instance that HeavyIQ is intended to communicate with. As HeavyDB is used in many industries for a wide variety of purposes, it is prohibitively expensive to train a LLM for each instance; instead, the information a LLM needs to effectively communicate with HeavyDB must be provided in the prompt itself.

Unfortunately, a current limitation of LLMs is a limit to the prompt size that can be parsed and kept “in context” by the LLM as it formulates its response (this is known as a “token limit”). As there is no limit to the quantity or complexity of data stored within HeavyDB, it is not possible to provide a full description of each table schema in every prompt. This is one of the main obstacles HeavyIQ is designed to overcome—providing the LLM with the proper context required to respond.



## System Overview

![Quick Architecture Diagram](../media/architecture.png?raw=true)

A HeavyDB Metadata Index (`HeavyDBMetadataIndex`) is generated and updated to hold the details and overviews of the tables present in the HeavyDB instance. This index aids in obtaining and delivering the required context for LLM prompts. The HeavyDB Metadata Index is searched to identify relevant tables, allowing the table structure to be supplied to the LLM to produce valid SQL.

HeavyIQ utilizes [LangChain](https://python.langchain.com/en/latest/index.html), a framework for developing applications powered by LLMs, for [prompt templating](https://python.langchain.com/en/latest/modules/prompts.html), [LLM output parsing](https://python.langchain.com/en/latest/modules/prompts/output_parsers.html), [performing operations on documents](https://python.langchain.com/en/latest/modules/chains/index_examples/qa_with_sources.html) , and more. LangChain provides useful building blocks that allow developers to establish boundaries for LLMs such that they can be integrated more deterministically alongside more traditional classes and functions. LangChain also rests on the cutting edge of LLM integration, granting LLMs the ability to  [act as an agent](https://python.langchain.com/en/latest/modules/agents.html) capable of ‘Reasoning & Acting’ ([ReAct](https://arxiv.org/pdf/2210.03629.pdf)). If one were to create a clone of ChatGPT w/ Plugins, it could be [accomplished almost entirely with LangChain alone](https://python.langchain.com/en/latest/modules/agents/agents/custom_llm_chat_agent.html).


## Interfaces

### REST API

HeavyIQ employs the [connexion module](https://pypi.org/project/connexion/) to offer a web communication interface that complies with the [OpenAPI 3.0 Specification](https://github.com/OAI/OpenAPI-Specification/blob/main/versions/3.0.3.md). When running HeavyIQ locally, the auto-generated API documentation can be accessed at http://127.0.0.1:5000/api/v1/ui/. Adhering to the OpenAPI 3.0 Specification is crucial as LLMs functioning as Agents with the capability to **Re**ason and **Act** can interpret OpenAPI 3.0 specifications and utilize various REST endpoints to execute more intricate behavior in order to achieve user-defined objectives.

### Command-Line Interface

HeavyIQ utilizes the [click module](https://pypi.org/project/click/) to allow interactions with HeavyIQ modules.


## Key Components

### HeavyDB Class

This class is a wrapper for interacting with a HeavyDB instance. It is responsible for access control, retrieving table schemas, sample rows, and validating and executing SQL statements against the HeavyDB instance.

### HeavyDBMetadataIndex Class

This class is a wrapper for interacting with the HeavyDB Metadata Index. Under the hood, the metadata index is an embeddings database (a.k.a. vector database), which stores embeddings generated from documents describing the tables and data found within the HeavyDB instance. Embeddings are important because they allow you to store words by their semantic meaning, such that if you store the word “ship” as an embedding, if you were to query the index with the word “boat”, the record containing “ship” would be returned as they are semantically similar to one another.

### Chains

Using an LLM in isolation is fine for some simple applications, but many more complex ones require chaining LLMs together and using the response of one LLM in the prompt of another. [LangChain Documentation](https://python.langchain.com/en/latest/modules/chains.html)

### Agents

Some applications will require not just a predetermined chain of calls to LLMs, but potentially an unknown chain that depends on the user’s input. In these situations, there is a “agent” which has access to a suite of tools and the ability to Reason & Act. Depending on the user input, the agent can then decide which, if any, of these tools to utilize to accomplish the user input. [LangChain Documentation](https://python.langchain.com/en/latest/modules/agents.html)

## Security

### Table-Level Access Permissions

HeavyIQ ensures the security and privacy of data stored in HeavyDB by implementing table-level access permissions in two ways:

1. **Session Authentication Level with HeavyDB**: When a user interacts with HeavyDB through HeavyIQ, their access permissions are checked at the session authentication level. This prevents users from seeing or querying tables they do not have the necessary permissions to access. HeavyDB handles this by ensuring that users can only interact with tables they are authorized to see via the typical roles/permissions.

2. **Prefiltering Metadata Index Queries**: To further secure the data and prevent unauthorized access, HeavyIQ pre-filters all user queries on the metadata index. This is done by only including documents for tables the user has permission to access. By prefiltering queries, HeavyIQ ensures that users cannot query the metadata index for tables they are not authorized to see, and therefore LLMs will not attempt to create SQL statements that can not be executed by the user.

## Logging

Calls to Chains and Agents are stored in a SQLite database found in the root of the application. This database captures:
* the type of call that was made (agent or chain)
* the name of the agent/chain called
* timestamps at the start and completion of the call
* the input arguments and resulting output
* any errors that resulted from the call
* token data (tokens used, cost of the call, etc)


## Conclusion

In summary, HeavyIQ is a powerful and versatile Python module that leverages large language models to provide a natural language interface for interacting with HeavyDB. By utilizing the HeavyDB Metadata Index and LangChain framework, HeavyIQ overcomes the limitations of LLMs and enables users to seamlessly communicate with their HeavyDB instances. With support for REST API and Command-Line Interface, HeavyIQ offers a flexible and user-friendly solution for a wide range of applications. As the field of natural language processing continues to advance, HeavyIQ is poised to further enhance the user experience and streamline the process of gaining insights from data stored in HeavyDB.
