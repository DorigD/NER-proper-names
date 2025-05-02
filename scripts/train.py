import os
import json
import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    DataCollatorForTokenClassification,

)
import torch
from datasets import load_from_disk
import gc
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils.metrics import compute_metrics, compute_entity_level_metrics, extract_entities
from utils.config import LABELS, NUM_LABELS, MODEL_NAME, BEST_PARAMS_PATH, DATASET_PATH, MODEL_OUTPUT_DIR, PROJECT_DIR 
from datetime import datetime

def get_next_version_number(base_dir):
    """
    Get the next version number by scanning existing version directories.
    
    Args:
        base_dir (str): Base directory to scan for version folders
        
    Returns:
        int: Next version number (1 if no versions exist yet)
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return 1
        
    # Look for directories with pattern "version-X"
    versions = []
    for item in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, item)) and item.startswith("version-"):
            try:
                version_num = int(item.split("-")[1])
                versions.append(version_num)
            except (IndexError, ValueError):
                # Skip if the format doesn't match our expected pattern
                continue
    
    # Return next version number (or 1 if no versions exist)
    return max(versions, default=0) + 1

def train_model(model_name=MODEL_NAME):
    
    tokenized_dataset = load_from_disk(DATASET_PATH)
    print(f"Dataset loaded: {tokenized_dataset}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare model and explicitly move to GPU
    print(f"Loading base model: {model_name}")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=NUM_LABELS
    )
    if torch.cuda.is_available():
        model = model.cuda()  # Use this instead of .to(device)
    
    # Verify model placement
    print(f"Model device: {next(model.parameters()).device}")

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
            logging_dir=os.path.join(PROJECT_DIR, "logs", "train"),
            logging_steps=10,
            eval_strategy="epoch", 
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="person_entity_f1",
            greater_is_better=True,
            no_cuda=False,
            fp16=torch.cuda.is_available(), 
            dataloader_num_workers=4,
        )

    # Early in the script, try to load best hyperparameters
    best_params = {
        "num_train_epochs": 5,
        "per_device_train_batch_size": 16,
        "weight_decay": 0.01
    }

    try:
        if os.path.exists(BEST_PARAMS_PATH):
            with open(BEST_PARAMS_PATH, "r") as f:
                best_params = json.load(f)
            print(f"Loaded optimized hyperparameters: {best_params}")
        else:
            print("Using default hyperparameters (no optimization file found)")
    except Exception as e:
        print(f"Error loading hyperparameters: {e}, using defaults")

    # Create models base directory if it doesn't exist
    models_base_dir = os.path.dirname(MODEL_OUTPUT_DIR)
    os.makedirs(models_base_dir, exist_ok=True)
    
    # Get next version number
    next_version = get_next_version_number(models_base_dir)
    versioned_output_dir = os.path.join(models_base_dir, f"version-{next_version}-{MODEL_NAME}")
    
    print(f"Training model version: {next_version}")

    # Create a single training run with the best available parameters
    training_args = create_training_arguments(
        output_dir=versioned_output_dir,
        num_train_epochs=best_params["num_train_epochs"],
        per_device_train_batch_size=best_params["per_device_train_batch_size"],
        weight_decay=best_params["weight_decay"],
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,  # Fixed parameter name
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],  # Use validation set during training
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Define the path for saving training results
    TRAINING_RESULTS_PATH = os.path.join(PROJECT_DIR, "logs", "training_results.json")

    # Train the model
    print("Starting training...")
    trainer.train()

    # Evaluate the model on validation set
    print("Evaluating model on validation set...")
    eval_results = trainer.evaluate()
    print(f"Validation results: {eval_results}")
    
    # Also evaluate on test set for final assessment
    print("Evaluating model on test set...")
    test_results = trainer.evaluate(tokenized_dataset["test"])
    print(f"Test results: {test_results}")

    # Save evaluation results to the logs folder
    os.makedirs(os.path.dirname(TRAINING_RESULTS_PATH), exist_ok=True)
    with open(TRAINING_RESULTS_PATH, "w") as f:
        json.dump({
            "validation": eval_results,
            "test": test_results
        }, f, indent=2)
    print(f"Training results saved to {TRAINING_RESULTS_PATH}")

    # Before saving model
    # Clear cache to free up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()  # Force garbage collection

    # Save the model to versioned directory
    trainer.save_model(versioned_output_dir)
    print(f"Model saved to {versioned_output_dir}")

    # Save model configuration with label mappings
    model_config = {
        "base_model": model_name,
        "labels": LABELS,
        "id2label": {str(i): label for label, i in LABELS.items()},
        "label2id": LABELS,
        "version": next_version
    }

    # Save with the expected filename for load_model compatibility
    with open(os.path.join(versioned_output_dir, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=2)
        
    # Copy the versioned model to the standard path for easy access
    import shutil
    if os.path.exists(MODEL_OUTPUT_DIR):
        if os.path.isdir(MODEL_OUTPUT_DIR):
            shutil.rmtree(MODEL_OUTPUT_DIR)
        else:
            os.remove(MODEL_OUTPUT_DIR)
    
    shutil.copytree(versioned_output_dir, MODEL_OUTPUT_DIR)
    print(f"Model copied to standard location: {MODEL_OUTPUT_DIR}")
    
    # Create a version_info.json file at the standard location to track current version
    with open(os.path.join(models_base_dir, "version_info.json"), "w") as f:
        json.dump({
            "current_version": next_version,
            "latest_model_path": versioned_output_dir,
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    
    print("Training completed successfully!")
    return versioned_output_dir

