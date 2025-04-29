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
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)
DATASET_PATH = os.path.join(PROJECT_DIR, "data", "tokenized_train")
MODEL_OUTPUT_DIR = os.path.join(PROJECT_DIR, "models", "roberta-finetuned-ner")

# Ensure the model output directory exists
if not os.path.exists(MODEL_OUTPUT_DIR):
    os.makedirs(MODEL_OUTPUT_DIR)

# Constants
MODEL_NAME = "roberta-base"
NUM_LABELS = 4  # O, B-PERSON, I-PERSON, TITLE
LABELS = {"O": 0, "B-PERSON": 1, "I-PERSON": 2, "TITLE": 3}


# Load tokenized dataset
print("Loading tokenized dataset...")
tokenized_dataset = load_from_disk(DATASET_PATH)

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
    # Add learning rate as a hyperparameter
    num_train_epochs = trial.suggest_int("num_train_epochs", 3, 10)
    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32])
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.3)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)

    # Use a unique output directory for each trial
    trial_output_dir = os.path.join(MODEL_OUTPUT_DIR, f"trial-{trial.number}")
    if not os.path.exists(trial_output_dir):
        os.makedirs(trial_output_dir)

    # Prepare model
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=trial_output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        weight_decay=weight_decay,
        learning_rate=learning_rate,  # Added learning rate
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(trial_output_dir, "logs"),
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        # Add gradient clipping to prevent numerical instability
        max_grad_norm=1.0,
        # Start with smaller batches for evaluation
        per_device_eval_batch_size=8,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],  
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
    # Enable device-side assertions for better error messages
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    try:
        study = optuna.create_study(direction="maximize",
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0))
        study.optimize(objective, n_trials=16, n_jobs=1)
        
        # Save the best parameters
        os.makedirs(os.path.join(PROJECT_DIR, "logs"), exist_ok=True)
        with open(os.path.join(PROJECT_DIR, "logs", "optuna_study_results.json"), "w") as f:
            json.dump(study.best_params, f, indent=2)
        
        print("Best hyperparameters: ", study.best_params)
    except Exception as e:
        print(f"Optimization failed with error: {e}")