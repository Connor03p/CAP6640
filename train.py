import os
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer, Trainer
from datasets import load_from_disk, Dataset
import transformers
from spy import Transformer

language = "py"
max_examples = 1000
transformer = Transformer()

model_path = "Salesforce/codegen-350M-multi"
output_path = "/media/Projects/In Progress/SimPy-master/checkpoints/" + language

model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir="./cached", local_files_only=True, device_map="auto")
dataset = load_from_disk("./cached/flytech_py")
dataset = dataset.select(range(max_examples))

data = []

for example in dataset:    
    data.append({
        "prompt": example["prompt"], 
        "code": example["code"]
    })
dataset = Dataset.from_list(data)

# Tokenize the dataset
tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir="./cached")
tokenizer.pad_token = tokenizer.eos_token
if language == 'spy':
        tokenizer.add_tokens(transformer.special_tokens)
def preprocess(examples):
    full_text = [p + c for p, c in zip(examples["prompt"], examples["code"])]
    tokenized = tokenizer(
        full_text,
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized
tokenized = dataset.map(preprocess, batched=True, remove_columns=["prompt", "code"])

# Create training and testing sets
tokenized = tokenized.train_test_split(test_size=0.1, shuffle=True)
print("Train: " + str(len(tokenized["train"])) + ", Test: " + str(len(tokenized["test"])))

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=1.8e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    fp16=True,
    weight_decay=0.1,
    num_train_epochs=1,
    save_strategy="epoch",
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
