import json

# List to store the data
data = []

# Open the JSON lines file
with open('arxiv-metadata-oai-snapshot.json', 'r') as f:
    for i, line in enumerate(f):
        # Stop after 100000 lines
        print("data processed: {0}".format(i))
        if i == 100000:
            break
        # Parse the JSON line and append to the list
        data.append(json.loads(line))

# Write out as a JSON array
with open('prepared_data.json', 'w') as f:
    json.dump(data, f, indent=4)
