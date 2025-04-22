import os
import timeit
import numpy as np
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer, Trainer
from datasets import load_from_disk, Dataset
import transformers
from spy import Transformer
import matplotlib.pyplot as plt

language = "spy"
num_tests = 9999999


dataset = load_from_disk("./cached/flytech_spy")
if num_tests > len(dataset):
    num_tests = len(dataset)

model_path = "Salesforce/codegen-350M-multi"
transformer = Transformer()

def convert_to_spy(sample):
    try:
        spy_code = transformer.parse(sample)
        return spy_code
    except (ValueError, RecursionError):
        return None
    
def convert_to_py(sample):
    try:
        spy_code = transformer.decode(sample)
        return spy_code
    except (ValueError, RecursionError):
        return None
    
# Token counting
tokenizer_py = AutoTokenizer.from_pretrained(model_path, cache_dir="./cached")
tokenizer_py.pad_token = tokenizer_py.eos_token
tokenizer_spy = AutoTokenizer.from_pretrained(model_path, cache_dir="./cached")
tokenizer_spy.pad_token = tokenizer_spy.eos_token
if language == 'spy':
        tokenizer_spy.add_tokens(transformer.special_tokens)

def token_count(example):
    full_text = example
    if language == "spy": tokenized = tokenizer_spy(full_text, padding="max_length", truncation=True, max_length=512)
    elif language == "py": tokenized = tokenizer_py(full_text, padding="max_length", truncation=True, max_length=512)
    return len([val for val in tokenized["attention_mask"] if val == 1])
    

transformer = Transformer()
total_time = 0
x = []
y = []


for i in range(num_tests):
    sample = dataset[i]["code"]
    time_start = timeit.default_timer()
    if language == "spy": convert_to_py(sample)
    elif language == "py": convert_to_spy(sample)
    time_end = timeit.default_timer()
    total_time += time_end - time_start
    tokens = token_count(sample)
    x.append(tokens)
    y.append(time_end - time_start)


print("Total Time: " + str(total_time))
print("Num Tests: " + str(num_tests))
print("Average Time: " + str(total_time / num_tests))

plt.scatter(x, y, s=1)

#calculate equation for trendline
z = np.polyfit(x, y, 1)
p = np.poly1d(z)

plt.plot(x, p(x), color="red")
plt.show()

