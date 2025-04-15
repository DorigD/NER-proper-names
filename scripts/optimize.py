import os
import json
import numpy as np
import optuna
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    EvalPrediction,
)
from datasets import load_from_disk
from sklearn.metrics import precision_recall_fscore_support

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
dataset_path = os.path.join(project_dir, "data", "tokenized_train")
model_output_dir = os.path.join(project_dir, "models", "roberta-finetuned-ner")

# Ensure the model output directory exists
if not os.path.exists(model_output_dir):
    os.makedirs(model_output_dir)

# Constants
MODEL_NAME = "roberta-base"
NUM_LABELS = 3  # O, B-PERSON, I-PERSON
LABELS = {"O": 0, "B-PERSON": 1, "I-PERSON": 2}

# Load tokenized dataset
print("Loading tokenized dataset...")
tokenized_dataset = load_from_disk(dataset_path)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def compute_metrics(eval_pred: EvalPrediction):
    # Ensure the function accepts a single EvalPrediction object
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    true_predictions = np.argmax(predictions, axis=2)
    true_labels = labels

    # Remove ignored index (special tokens)
    true_predictions = [
        [p for (p, l) in zip(pred, label) if l != -100] 
        for pred, label in zip(true_predictions, true_labels)
    ]
    true_labels = [
        [l for l in label if l != -100]
        for label in true_labels
    ]

    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        [l for sublist in true_labels for l in sublist],
        [p for sublist in true_predictions for p in sublist],
        average="micro"
    )
    
    return {"f1": f1_micro}

def objective(trial):
    # Define hyperparameters to optimize
    num_train_epochs = trial.suggest_int("num_train_epochs", 3, 10)
    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32])
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.3)

    # Use a unique output directory for each trial
    trial_output_dir = os.path.join(model_output_dir, f"trial-{trial.number}")
    if not os.path.exists(trial_output_dir):
        os.makedirs(trial_output_dir)

    # Prepare model
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=trial_output_dir,  # Use trial-specific output directory
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        weight_decay=weight_decay,
        evaluation_strategy="epoch",  # Ensure evaluation is done per epoch
        save_strategy="epoch",        # Match save strategy with evaluation strategy
        logging_dir=os.path.join(trial_output_dir, "logs"),
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],  # Use validation set instead of test
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    return eval_results["eval_f1"]

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    # Save the best parameters
    with open(os.path.join(project_dir, "logs", "optuna_study_results.json"), "w") as f:
        json.dump(study.best_params, f, indent=2)

    print("Best hyperparameters: ", study.best_params)