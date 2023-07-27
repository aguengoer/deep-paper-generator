import json
import random

# List to store the data
data = []

# Open the JSON lines file
with open('arxiv-metadata-oai-snapshot.json', 'r') as f:
    # Parse all lines into a list
    for i, line in enumerate(f):
        # Stop after 100000 lines
        print("data processed: {0}".format(i))

        if i == 1000000:
            break
        # Parse the JSON line and append to the list
        data.append(json.loads(line))

# Select 100 random entries
data = random.sample(data, 1000)

# Write out as a JSON array
with open('random_test_data_100.json', 'w') as f:
    json.dump(data, f, indent=4)
