from datasets import load_from_disk
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline


model_path = "/media/Projects/In Progress/SimPy-master/checkpoints/py/"  # or use your model's identifier if uploaded to Hugging Face Hub
token_path = "/media/Projects/In Progress/SimPy-master/checkpoints/py/"
tokenizer = AutoTokenizer.from_pretrained(token_path, cache_dir="./cached", local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
dataset = load_from_disk("./cached/flytech_original")

task = "text-generation"  # Change based on your model type
generator = pipeline(task, model=model, tokenizer=tokenizer)

task = dataset[2]["instruction"]
print("Task: " + task)
print(generator(task, max_length=300))
