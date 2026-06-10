# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import heavyai
import numpy as np
import pandas as pd
import sys
import argparse


class SentenceTransformerEmbeddingFunction:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ValueError(
                "The sentence_transformers python package is not installed. Please install it with `pip install sentence_transformers`"
            )
        self._model = SentenceTransformer(model_name)

    def __call__(self, texts):
        return self._model.encode(texts, convert_to_numpy=True)


def main(argv):
    # Set up command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", type=str, default="admin")
    parser.add_argument("--dbname", type=str, default="heavyai")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--password", type=str, default="HyperInteractive")
    parser.add_argument("--model_name", type=str, default="gtr-t5-large")
    parser.add_argument("--needle_table", type=str, default="search_needle")
    parser.add_argument("--min_dot_product", type=float, default=0.6)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--source_table", type=str, required=True)
    parser.add_argument("--embedding_column", type=str, required=True)
    parser.add_argument("--text_column", type=str, required=True)

    args = parser.parse_args()

    # Connect to the database
    con = heavyai.connect(user=args.user, dbname=args.dbname, host=args.host, password=args.password)
    model = SentenceTransformerEmbeddingFunction(args.model_name)

    while True:
        search_input = input("Search query: ")
        if search_input == "quit":
            break
        model_output = model(search_input)
        embedding_size = model_output.shape[0]
        embedding = pd.DataFrame([[model_output.tolist()]], columns=["embedding"])

        con.execute(f"DROP TABLE IF EXISTS {args.needle_table}")
        con.execute(f"CREATE TABLE {args.needle_table} (embedding FLOAT[{embedding_size}])")
        con.load_table_columnar(args.needle_table, embedding)

        results = list(
            con.execute(
                f"SELECT a.{args.text_column}, DOT_PRODUCT(a.{args.embedding_column}, b.embedding) AS score FROM {args.source_table} a, {args.needle_table} b WHERE DOT_PRODUCT(a.{args.embedding_column}, b.embedding) > {args.min_dot_product} ORDER BY DOT_PRODUCT(a.{args.embedding_column}, b.embedding) DESC LIMIT {args.top_k}"
            )
        )

        for idx, result in enumerate(results, start=1):
            print("""{idx}. {text}: {score}""".format(idx=idx, text=result[0], score=result[1]))

        print("\n")


if __name__ == "__main__":
    main(sys.argv[1:])
