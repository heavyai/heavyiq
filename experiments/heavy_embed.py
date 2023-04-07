import heavyai
import pandas as pd
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


# Set up command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--user", type=str, default="admin")
parser.add_argument("--dbname", type=str, default="heavyai")
parser.add_argument("--host", type=str, default="127.0.0.1")
parser.add_argument("--password", type=str, default="HyperInteractive")
parser.add_argument("--model_name", type=str, default="gtr-t5-large")
parser.add_argument("--source_table", type=str, required=True)
parser.add_argument("--id_column", type=str, required=True)
parser.add_argument("--text_column", type=str, required=True)
parser.add_argument("--id_filter", type=str, required=True)
parser.add_argument("--destination_table", type=str, required=True)

args = parser.parse_args()

# Connect to the database
con = heavyai.connect(user=args.user, dbname=args.dbname, host=args.host, password=args.password)

# Initialize the SentenceTransformer model
model = SentenceTransformerEmbeddingFunction(args.model_name)

# Fetch data from the source table
query = f"SELECT {args.id_column}, {args.text_column} FROM {args.source_table} WHERE {args.id_filter}"
source_data = con.select_ipc(query)
ids = list(source_data[args.id_column])
strs = list(source_data[args.text_column])

# Compute embeddings
embeddings = model(strs)
embedding_size = embeddings.shape[1]

# Create the DataFrame
df = pd.DataFrame({"id": ids, "str": strs, "embedding": embeddings.tolist()})

# Create the destination table if it doesn't exist
con.execute(
    f"CREATE TABLE IF NOT EXISTS {args.destination_table} (id BIGINT, str TEXT ENCODING NONE, embedding FLOAT[{embedding_size}])"
)

# Load the DataFrame into the destination table
con.load_table_columnar(args.destination_table, df)
