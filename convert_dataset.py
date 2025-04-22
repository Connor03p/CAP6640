from datasets import load_from_disk
import re
from transformers import AutoTokenizer
from datasets import Dataset
from spy import Transformer

config = {
    "dataset_dir": "./cached/flytech_original",
    "code_column_name": "output",
    "lang": "py",
    "prompt_column_name": "instruction",
    "cache_dir": "./cached",
    "save_dir": "./cached/flytech_py"
}

dataset = load_from_disk("./cached/flytech_original")

def trim(sample):
    first_line = sample.partition('\n')[0]

    if (first_line == '```python'):
        sample = sample[10:]
        sample = sample[:(len(sample) - 3)]
    
    elif (first_line == '"""'):
        sample = sample[3:]
        sample = sample[:(len(sample) - 3)]

    return sample

# Conversion to SimPy
transformer = Transformer()
def convert_to_spy(sample):
    output = trim(sample)

    try:
        spy_code = transformer.parse(output)
        return spy_code
    except (ValueError, RecursionError):
        return None
    

# Iterate through dataset
data = []
for example in dataset:
    if example["output"] == "":
        continue

    code = convert_to_spy(example["output"])
    if code == None: continue

    if config["lang"] == "py":
        code = trim(example["output"])

    data.append({
        "prompt": example["instruction"], 
        "code": code
    })
dataset = Dataset.from_list(data)

# Print results
print(dataset)
for example in dataset:
    print("---------------------------------\n" + example["code"])
print("Length: " + str(len(dataset)))

# Save converted dataset
dataset.save_to_disk(config["save_dir"])