import json
import random

# Load the JSON data
with open("sql_training_prompts.json") as file:
    data = json.load(file)

# Randomize the order of the JSON objects
random.shuffle(data)

# Save the results
with open("sql_training_prompts_randomized.json", "w") as file:
    json.dump(data, file, indent=4)
