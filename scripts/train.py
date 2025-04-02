import os
import json
import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EvalPrediction
)
from datasets import load_from_disk
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import optuna

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
dataset_path = os.path.join(project_dir, "data", "tokenized_train")
model_output_dir = os.path.join(project_dir, "models", "roberta-finetuned-ner")

# Constants
MODEL_NAME = "roberta-base"
NUM_LABELS = 3  # O, B-PERSON, I-PERSON
LABELS = {"O": 0, "B-PERSON": 1, "I-PERSON": 2}

# Create output directory if it doesn't exist
os.makedirs(model_output_dir, exist_ok=True)

# Load tokenized dataset
print("Loading tokenized dataset...")
tokenized_dataset = load_from_disk(dataset_path)
print(f"Dataset loaded: {tokenized_dataset}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Prepare model
print(f"Loading base model: {MODEL_NAME}")
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS
)

# Define metrics computation function
def compute_metrics(p: EvalPrediction):
    predictions = np.argmax(p.predictions, axis=2)
    labels = p.label_ids
    
    # Remove ignored index (special tokens)
    true_predictions = [
        [p for (p, l) in zip(pred, label) if l != -100] 
        for pred, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for l in label if l != -100]
        for label in labels
    ]
    
    # Token-level metrics
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        [l for sublist in true_labels for l in sublist],
        [p for sublist in true_predictions for p in sublist],
        average="micro"
    )
    
    # Separate metrics for B-PERSON and I-PERSON
    b_person_metrics = precision_recall_fscore_support(
        [l for sublist in true_labels for l in sublist],
        [p for sublist in true_predictions for p in sublist],
        labels=[1],  # B-PERSON only
        average="micro"  # Changed from "binary" to "micro"
    )
    
    i_person_metrics = precision_recall_fscore_support(
        [l for sublist in true_labels for l in sublist],
        [p for sublist in true_predictions for p in sublist],
        labels=[2],  # I-PERSON only
        average="micro"  # Changed from "binary" to "micro"
    )
    
    # Combined PERSON entity metrics
    person_metrics = precision_recall_fscore_support(
        [l for sublist in true_labels for l in sublist],
        [p for sublist in true_predictions for p in sublist],
        labels=[1, 2],  # Both B-PERSON and I-PERSON
        average="micro"
    )
    
    # Entity-level span evaluation (more realistic evaluation)
    entity_results = compute_entity_level_metrics(true_labels, true_predictions)
    
    accuracy = accuracy_score(
        [l for sublist in true_labels for l in sublist],
        [p for sublist in true_predictions for p in sublist]
    )
    
    return {
        "accuracy": accuracy,
        "token_precision": precision_micro,
        "token_recall": recall_micro,
        "token_f1": f1_micro,
        "person_precision": person_metrics[0],
        "person_recall": person_metrics[1],
        "person_f1": person_metrics[2],
        "b_person_f1": b_person_metrics[2],
        "i_person_f1": i_person_metrics[2],
        "entity_precision": entity_results["precision"],
        "entity_recall": entity_results["recall"],
        "eval_person_f1": entity_results["f1"],  # Using entity-level F1 as main metric
    }

def compute_entity_level_metrics(true_labels, true_predictions):
    """
    Compute entity-level metrics by extracting whole name spans instead of individual tokens.
    This better measures if the model correctly identifies complete names.
    """
    true_entities = []
    pred_entities = []
    
    # Extract entity spans from labels
    for doc_labels, doc_preds in zip(true_labels, true_predictions):
        true_doc_entities = extract_entities(doc_labels)
        pred_doc_entities = extract_entities(doc_preds)
        
        true_entities.extend(true_doc_entities)
        pred_entities.extend(pred_doc_entities)
    
    # Calculate metrics
    correct = len(set(true_entities) & set(pred_entities))
    precision = correct / len(pred_entities) if pred_entities else 0
    recall = correct / len(true_entities) if true_entities else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1": f1}

def extract_entities(token_labels):
    """
    Extract entity spans from token-level predictions.
    Returns a list of tuples (start_idx, end_idx, entity_type).
    """
    entities = []
    entity_start = None
    
    for i, label in enumerate(token_labels):
        if label == 1:  # B-PERSON
            if entity_start is not None:
                entities.append((entity_start, i-1, "PERSON"))
            entity_start = i
        elif label == 2:  # I-PERSON
            continue
        elif entity_start is not None:
            entities.append((entity_start, i-1, "PERSON"))
            entity_start = None
    
    # Handle entity at the end of sequence
    if entity_start is not None:
        entities.append((entity_start, len(token_labels)-1, "PERSON"))
        
    return entities

# Prepare data collator
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# Function to create training arguments
def create_training_arguments(output_dir, num_train_epochs, per_device_train_batch_size, weight_decay):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=weight_decay,
        logging_dir=os.path.join(project_dir, "logs"),
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_person_f1",
        greater_is_better=True,
    )

# Early in the script, try to load best hyperparameters
best_params = {
    "num_train_epochs": 5,
    "per_device_train_batch_size": 16,
    "weight_decay": 0.01
}

try:
    best_params_path = os.path.join(project_dir, "logs", "optuna_study_results.json")
    if os.path.exists(best_params_path):
        with open(best_params_path, "r") as f:
            best_params = json.load(f)
        print(f"Loaded optimized hyperparameters: {best_params}")
    else:
        print("Using default hyperparameters (no optimization file found)")
except Exception as e:
    print(f"Error loading hyperparameters: {e}, using defaults")

# Create a single training run with the best available parameters
training_args = create_training_arguments(
    output_dir=model_output_dir,
    num_train_epochs=best_params["num_train_epochs"],
    per_device_train_batch_size=best_params["per_device_train_batch_size"],
    weight_decay=best_params["weight_decay"],
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Define the path for saving training results
training_results_path = os.path.join(project_dir, "logs", "training_results.json")

# Train the model
print("Starting training...")
trainer.train()

# Evaluate the model
print("Evaluating model...")
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# Save evaluation results to the logs folder
os.makedirs(os.path.dirname(training_results_path), exist_ok=True)
with open(training_results_path, "w") as f:
    json.dump(eval_results, f, indent=2)
print(f"Training results saved to {training_results_path}")

# Save the model
trainer.save_model(model_output_dir)
print(f"Model saved to {model_output_dir}")

# Save model configuration with label mappings for later reference
model_config = {
    "base_model": MODEL_NAME,
    "labels": LABELS,
    "id2label": {str(i): label for label, i in LABELS.items()},
    "label2id": LABELS,
}

with open(os.path.join(model_output_dir, "model_config.json"), "w") as f:
    json.dump(model_config, f, indent=2)

print("Training completed successfully!")