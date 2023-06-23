import pandas as pd

# Load the CSV file
df = pd.read_csv("spider_qa_modified.csv")

# Replace double quotes with single quotes in the 'original_sql_query' column
df["original_sql_query"] = (
    df["original_sql_query"].str.replace('""', "'", regex=False).str.replace('"', "'", regex=False)
)

# Save the modified dataframe to a new CSV file
df.to_csv("spider_qa_modified_single_quotes.csv", index=False)
