import timeit
from datasets import load_from_disk
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
import evaluate

num_tasks = 20
seed = 16

model_path = "/media/Projects/In Progress/SimPy-master/checkpoints/spy/"  # or use your model's identifier if uploaded to Hugging Face Hub
token_path = "/media/Projects/In Progress/SimPy-master/checkpoints/spy/"
tokenizer = AutoTokenizer.from_pretrained(token_path, cache_dir="./cached", local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
dataset = load_from_disk("./cached/flytech_spy")
dataset = dataset.shuffle(seed)
dataset = dataset.select(range(num_tasks))

task = "text-generation"  # Change based on your model type
generator = pipeline(task, model=model, tokenizer=tokenizer)

references = []
predictions = []

for i in range(num_tasks):
    sample = dataset[i]["code"]
    references.append([sample])

for i in range(num_tasks):
    problem = dataset[i]["prompt"]

    time_start = timeit.default_timer()
    result = generator(problem, max_length=300, truncation=True)
    time_end = timeit.default_timer()

    time_total = time_end - time_start
    result = result[0]["generated_text"]
    predictions.append(result)
    print("Finished task " + str(i) + " in " + str(time_total))
    
bleu = evaluate.load("bleu")
score = bleu.compute(predictions=predictions, references=references)
print(score)