# Architecture

## Introduction

HeavyNL is a Python module designed to provide a natural language interface to [HeavyDB](https://github.com/heavyai/heavydb) using large language models and embeddings databases (also known as vectorstores). The goal of HeavyNL is to use natural language processing to enable users of HeavyDB to interactively communicate with and gain insight into their data using their native language (as opposed to writing SQL or Python).

**HeavyNL is made available via three interfaces:**
* [OpenAPI 3.0 Specification](https://github.com/OAI/OpenAPI-Specification/blob/main/versions/3.0.3.md) Compliant REST API
* Command-line Interface
* Python module


## Background and Context

A large language model (or LLM) is an artificial intelligence system that is trained on vast amounts of text data to understand the nuances of human language. It uses this knowledge to generate human-like responses to natural language queries and tasks (a.k.a. ‘prompts’), such as answering questions or completing sentences.

LLMs, such as GPT-4, lack knowledge beyond the data they were trained on, hence they do not possess any inherent knowledge of the contents of the HeavyDB instance that HeavyNL is intended to communicate with. As HeavyDB is used in many industries for a wide variety of purposes, it is prohibitively expensive to train a LLM for each instance; instead, the information a LLM needs to effectively communicate with HeavyDB must be provided in the prompt itself.

Unfortunately, a current limitation of LLMs is a limit to the prompt size that can be parsed and kept “in context” by the LLM as it formulates its response (this is known as a “token limit”). As there is no limit to the quantity or complexity of data stored within HeavyDB, it is not possible to provide a full description of each table schema in every prompt. The is one of the main obstacles HeavyNL is designed to overcome—providing the LLM with the proper context required to respond.


## System Overview

A HeavyDB Metadata Index is generated and updated to hold the details and overviews of the tables present in the HeavyDB instance. This index aids in obtaining and delivering the required context for LLM prompts. The HeavyDB Metadata Index is searched to identify relevant tables, allowing the table structure to be supplied to the LLM to produce valid SQL.
