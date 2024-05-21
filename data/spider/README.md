## What is the Spider dataset?

Spider is a large-scale complex and cross-domain semantic parsing and text-to-SQL dataset annotated by 11 Yale students. The goal of the Spider challenge is to develop natural language interfaces to cross-domain databases. It consists of 10,181 questions and 5,693 unique complex SQL queries on 200 databases with multiple tables covering 138 different domains. In Spider 1.0, different complex SQL queries and databases appear in train and test sets. To do well on it, systems must generalize well to not only new SQL queries but also new database schemas.

## What is it doing here?

By transposing the SQLlite to HeavySQL we can use this dataset to fine-tune large language models (LLMs) to only generate valid HeavySQL. A fine-tuned LLM should produce better results than a general-use model; general-use models will often confuse SQL dialects, generating SQL statements that are invalid for execution on HeavyDBs.
