# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset (use your actual dataset)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Add a special padding token if it doesn't exist
tokenizer.pad_token = tokenizer.eos_token

# Tokenization function
def tokenize_function(examples):
    # Tokenize the inputs and set them as both input_ids and labels
    tokens = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
    tokens['labels'] = tokens['input_ids'].copy()
    return tokens

# Tokenize the dataset and include labels
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load model (make sure the model is capable of causal language modeling)
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

# Start training
trainer.train()
