import os
import timeit
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer, Trainer
from datasets import load_from_disk, Dataset
import transformers
from spy import Transformer

language = "spy"
transformer = Transformer()


model_path = "Salesforce/codegen-350M-multi"
dataset = load_from_disk("./cached/flytech_spy")

# Tokenize the dataset
tokenizer_py = AutoTokenizer.from_pretrained(model_path, cache_dir="./cached")
tokenizer_py.pad_token = tokenizer_py.eos_token
tokenizer_spy = AutoTokenizer.from_pretrained(model_path, cache_dir="./cached")
tokenizer_spy.pad_token = tokenizer_spy.eos_token
if language == 'spy':
        tokenizer_spy.add_tokens(transformer.special_tokens)

def token_count(example):
    full_text = example["code"]
    time_start = timeit.default_timer()
    if language == "spy": tokenized = tokenizer_spy(full_text, padding="max_length", truncation=True, max_length=512)
    elif language == "py": tokenized = tokenizer_py(full_text, padding="max_length", truncation=True, max_length=512)
    time_end = timeit.default_timer()
    return {"tokens": len([val for val in tokenized["attention_mask"] if val == 1]), "time": time_end - time_start}

tokens = 0
elapsed = 0

groups = {
    "0-50": [],
    "50-100": [],
    "100-150": [],
    "150-200": [],
    "200-250": [],
    "250-300": [],
    "300-350": [],
    "350-400": [],
    "400-450": [],
    "450-500": [],
}

for example in dataset:
    result = token_count(example)
    tokens += result["tokens"]
    elapsed += result["time"]
    
    if (result["tokens"] < 50):
        groups["0-50"].append(result["time"])
    elif (result["tokens"] < 100):
        groups["50-100"].append(result["time"])
    elif (result["tokens"] < 150):
        groups["100-150"].append(result["time"])
    elif (result["tokens"] < 200):
        groups["150-200"].append(result["time"])
    elif (result["tokens"] < 250):
        groups["200-250"].append(result["time"])
    elif (result["tokens"] < 300):
        groups["250-300"].append(result["time"])
    elif (result["tokens"] < 350):
        groups["300-350"].append(result["time"])
    elif (result["tokens"] < 400):
        groups["350-400"].append(result["time"])
    elif (result["tokens"] < 450):
        groups["400-450"].append(result["time"])
    elif (result["tokens"] < 500):
        groups["450-500"].append(result["time"])
    
    
avg_tokens = tokens / len(dataset)
avg_elapsed = elapsed / len(dataset)
print("Tokens: " + str(tokens))
print("Elapsed: " + str(elapsed))
print("Examples: " + str(len(dataset)))
print("Average Tokens: " + str(avg_tokens))
print("Average Elapsed: " + str(avg_elapsed))

for key in groups:
    total = 0
    for value in groups[key]:
        total += value
    
    out = total / len(groups[key])
    print("Group " + key + " (" + str(len(groups[key])) + " items): " + str(out))


