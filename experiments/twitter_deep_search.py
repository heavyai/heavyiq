import heavyai
import numpy as np
import pandas as pd
import sys


class SentenceTransformerEmbeddingFunction:
    # If you have a beefier machine, try "gtr-t5-large".
    # for a full list of options: https://huggingface.co/sentence-transformers, https://www.sbert.net/docs/pretrained_models.html
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
    con = heavyai.connect(user="admin", password="HyperInteractive", host="100.85.68.75", dbname="heavyai")
    model = SentenceTransformerEmbeddingFunction()
    while True:
        search_input = input("Search query: ")
        if search_input == "quit":
            break
        model_output = model(search_input)
        embedding = pd.DataFrame([[model_output.tolist()]], columns=["embedding"])
        con.execute("DROP TABLE IF EXISTS twitter_needle")
        con.execute("CREATE TABLE twitter_needle (embedding FLOAT[384])")
        con.load_table_columnar("twitter_needle", embedding)
        results = list(
            con.execute(
                "SELECT a.tweet, DOT_PRODUCT(a.embeddings, b.embedding) AS score FROM ca_tweet_embeddings_full a, twitter_needle b WHERE DOT_PRODUCT(a.embeddings, b.embedding) > 0.6 ORDER BY DOT_PRODUCT(a.embeddings, b.embedding) DESC LIMIT 10"
            )
        )
        for result in results:
            print("""{tweet}: {score}""".format(tweet=result[0], score=result[1]))
        print("\n")


if __name__ == "__main__":
    main(sys.argv[1:])
