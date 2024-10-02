import importlib.util
import json
import time
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
# Import training and encryption functions
spec = importlib.util.spec_from_file_location("train", "model/gpt2.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)

# Load dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Tokenization function
def tokenize_function(examples):
    tokens = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
    tokens['labels'] = tokens['input_ids'].copy()
    return tokens

# Tokenize the dataset and include labels
tokenized_datasets = dataset.map(tokenize_function, batched=True)



# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # Change to `eval_strategy` to avoid deprecation warning
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Perform encryption
train.perform_encryption()

# Model training and evaluation
train.model_training(trainer, num_epochs=3, batch_size=10)
train.model_evaluation()




