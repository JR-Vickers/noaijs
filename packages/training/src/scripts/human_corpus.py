import json
import re

with open('../data/raw/human/casual/samples.json', 'r') as f:
    data = json.load(f)
with open('../data/raw/human/creative/samples.json', 'r') as f:
    data += json.load(f)
with open('../data/raw/human/essays/samples.json', 'r') as f:
    data += json.load(f)
with open('../data/raw/human/technical/samples.json', 'r') as f:
    data += json.load(f)

with open('../data/raw/human_corpus.txt', 'w') as out:
    for entry in data:
        text = re.sub(r'\s', ' ', entry['text']).strip()
        out.write(text + '\n')