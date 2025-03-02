import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from datasets import load_from_disk

# Define model and dataset paths
MODEL_NAME = "roberta-base"
DATASET_PATH = "data/tokenized_train"
SAVE_MODEL_PATH = "models/roberta-finetuned-ner"

# Load tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
dataset = load_from_disk(DATASET_PATH)

# Define label mappings (same as in preprocess.py)
LABELS = {"O": 0, "B-PERSON": 1, "I-PERSON": 2}
id2label = {v: k for k, v in LABELS.items()}
label2id = LABELS

# Load RoBERTa model for token classification
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME, num_labels=len(LABELS), id2label=id2label, label2id=label2id
)

# Define data collator for dynamic padding
data_collator = DataCollatorForTokenClassification(tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir=SAVE_MODEL_PATH,
    evaluation_strategy="epoch",  # Evaluate after each epoch
    save_strategy="epoch",        # Save model after each epoch
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,  # Adjust based on performance
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,  # Keep only the last 2 models
    push_to_hub=False,
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
trainer.train()

# Save final model
trainer.save_model(SAVE_MODEL_PATH)
tokenizer.save_pretrained(SAVE_MODEL_PATH)

print(f"Model fine-tuned and saved at {SAVE_MODEL_PATH}")
