import os
import json
import numpy as np
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer
)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from datasets import load_from_disk
import gc
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils.metrics import compute_metrics, compute_entity_level_metrics, extract_entities
from utils.config import LABELS, NUM_LABELS, MODEL_NAME, BEST_PARAMS_PATH, DATASET_PATH, MODEL_OUTPUT_DIR, PROJECT_DIR 
from datetime import datetime
from collections import Counter
import torch.nn as nn
from adapters import AdapterConfig, AutoAdapterModel
from adapters.composition import Stack
from transformers import EarlyStoppingCallback

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
    # Load hyperparameters FIRST, before using them
    default_params = {
        "num_train_epochs": 5,
        "per_device_train_batch_size": 16,
        "weight_decay": 0.01
    }

    best_params = default_params.copy()

    try:
        if os.path.exists(BEST_PARAMS_PATH):
            with open(BEST_PARAMS_PATH, "r") as f:
                loaded_data = json.load(f)
                
                # Check if parameters are nested under "best_params" key
                if "best_params" in loaded_data:
                    loaded_params = loaded_data["best_params"]
                else:
                    loaded_params = loaded_data
                
                # Update best_params with loaded values
                for key, value in loaded_params.items():
                    best_params[key] = value
        else:
            print("Using default hyperparameters (no optimization file found)")
    except Exception as e:
        print(f"Error loading hyperparameters: {e}, using defaults: {default_params}")

    # Make sure required keys exist
    required_keys = ["num_train_epochs", "per_device_train_batch_size", "weight_decay"]
    for key in required_keys:
        if key not in best_params:
            print(f"Warning: Missing required parameter '{key}', using default value: {default_params[key]}")
            best_params[key] = default_params[key]
    
    # Add to your hyperparameters section:
    gamma = 2.0  # Standard recommendation for imbalanced problems
    if "gamma" in best_params:
        gamma = best_params["gamma"]
    
    use_class_weights = best_params.get("use_class_weights", False)
    
    # THEN load dataset and model
    tokenized_dataset = load_from_disk(DATASET_PATH)
    print(f"--------------Dataset loaded: {tokenized_dataset}----------------")
    # Load tokenizer BEFORE model
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    # Use AutoAdapterModel instead of AutoModelForTokenClassification
    model = AutoAdapterModel.from_pretrained(model_name)
    
    # Add NER adapter
    # NER is a sequence tagging task, so we use a sequence classification adapter
    adapter_config = AdapterConfig.load(
        "pfeiffer",  # Efficient adapter architecture
        reduction_factor=16  # Controls adapter size (smaller=faster but less expressive)
    )
    model.add_adapter("ner_adapter", config=adapter_config)
    
    # Add NER classification head
    model.add_classification_head(
        "ner_adapter",
        num_labels=NUM_LABELS,
        id2label={str(i): label for label, i in LABELS.items()}
    )
    
    # Activate the adapter
    model.train_adapter("ner_adapter")
    
    # Freeze base model parameters
    model.freeze_model(True)
    
    if torch.cuda.is_available():
        model = model.cuda()

    # Set class weights AFTER model is loaded
    all_labels = []
    for item in tokenized_dataset["train"]:
        all_labels.extend([l for l in item["labels"] if l != -100])

    label_counts = Counter(all_labels)

    # Update to use person_weight from optimization if available
    person_weight = 5.0  # Default value
    if "person_weight" in best_params:
        person_weight = best_params["person_weight"]
  
    class_weights = torch.ones(NUM_LABELS)
    class_weights[0] = 0.2
    class_weights[1] = person_weight
    class_weights[2] = person_weight * best_params.get("i_person_ratio", 0.9)
    # Replace direct index access with this safer approach
    if "TITLE" in LABELS:
        title_idx = list(LABELS.values()).index(LABELS["TITLE"])
        class_weights[title_idx] = best_params.get("title_weight", 1.2)

    model.config.class_weights = class_weights.tolist()

    # Prepare data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Function to create training arguments
    def create_training_arguments(output_dir, training_params):
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=weight_decay,
            logging_strategy="steps",
            logging_steps=50,
            eval_strategy="epoch", 
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="person_entity_f1",
            greater_is_better=True,
            no_cuda=False,
            fp16=torch.cuda.is_available(), 
            dataloader_num_workers=4,
            learning_rate=5e-4,  # Higher learning rate for adapters
            lr_scheduler_type="cosine_with_restarts",  # Better scheduler
            warmup_ratio=0.1,
            save_total_limit=2,  # Keep only best 2 checkpoints
        )

    # Create models base directory if it doesn't exist
    models_base_dir = os.path.dirname(MODEL_OUTPUT_DIR)
    os.makedirs(models_base_dir, exist_ok=True)
    
    # Get next version number
    next_version = get_next_version_number(models_base_dir)
    versioned_output_dir = os.path.join(models_base_dir, f"version-{next_version}-{MODEL_NAME}")

    # Create a single training run with the best available parameters
    training_args = create_training_arguments(
        output_dir=versioned_output_dir,
        num_train_epochs=best_params["num_train_epochs"],
        per_device_train_batch_size=best_params["per_device_train_batch_size"],
        weight_decay=best_params["weight_decay"],
    )

    # Add before the trainer initialization

    # Create a focal loss trainer for better handling of imbalanced classes
    class FocalLossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            # Add adapter_names parameter to match main.py
            outputs = model(**inputs, adapter_names=["ner_adapter"])
            logits = outputs.logits
            
            # Parameters for focal loss
            gamma = 2.0  # Focusing parameter
            alpha = class_weights.to(model.device)
            
            # Compute probabilities
            probs = torch.softmax(logits.view(-1, NUM_LABELS), dim=-1)
            
            # Create one-hot encoding of labels
            active_loss = labels.view(-1) != -100
            active_labels = torch.where(active_loss, labels.view(-1), torch.tensor(0).type_as(labels))
            
            one_hot = torch.zeros_like(probs).scatter_(1, active_labels.unsqueeze(1), 1)
            
            # Compute focal weights
            pt = (one_hot * probs).sum(1) + 1e-10
            focal_weight = (1 - pt) ** gamma
            
            # Get class weights for each sample
            alpha_weight = torch.gather(alpha, 0, active_labels)
            
            # Compute focal loss
            loss = -alpha_weight * focal_weight * torch.log(pt)
            
            # Apply mask for padding
            loss = torch.where(active_loss, loss, torch.tensor(0.0).type_as(loss))
            
            loss = loss.mean()
            
            return (loss, outputs) if return_outputs else loss

    # Setup early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001
    )

    # Use FocalLossTrainer with early stopping
    trainer = FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping]  # Add early stopping callback
    )

    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))

    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Evaluate the model on validation set
    print("------Evaluating on validation set...------")
    eval_results = trainer.evaluate()

    test_results = trainer.evaluate(tokenized_dataset["test"])
    
    TRAINING_RESULTS_PATH = os.path.join(PROJECT_DIR, "logs", "training_results.json")

    # Create the current training result - MOVED HERE after eval_results and test_results exist
    current_result = {
        "version": next_version,
        "model_path": versioned_output_dir,
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hyperparameters": best_params,
        "class_weights": {i: float(w) for i, w in enumerate(class_weights.tolist())},
        "validation": eval_results,
        "test": test_results  
    }

    # Add to train.py evaluation section
    title_results = {
        "title_f1": test_results.get("eval_title_f1", 0.0),
        "title_precision": test_results.get("eval_title_precision", 0.0),
        "title_recall": test_results.get("eval_title_recall", 0.0)
    }
    current_result["title_metrics"] = title_results

    # Ensure directory exists
    os.makedirs(os.path.dirname(TRAINING_RESULTS_PATH), exist_ok=True)

    # Load existing results if file exists
    all_results = []
    if os.path.exists(TRAINING_RESULTS_PATH):
        try:
            with open(TRAINING_RESULTS_PATH, "r") as f:
                all_results = json.load(f)
            if not isinstance(all_results, list):
                # Convert old format to list if needed
                all_results = [all_results]
        except json.JSONDecodeError:
            print("Warning: Could not parse existing results file, creating new one")
            all_results = []

    # Append new results
    all_results.append(current_result)

    # Save updated results
    with open(TRAINING_RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Before saving model
    # Clear cache to free up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()  # Force garbage collection

    # Save the model to versioned directory
    trainer.save_model(versioned_output_dir)
  

    # Save adapter configuration for loading
    with open(os.path.join(versioned_output_dir, "adapter_config.json"), "w") as f:
        json.dump({
            "adapter_name": "ner_adapter",
            "base_model": model_name
        }, f, indent=2)
    
    # Save model configuration with label mappings
    model_config = {
        "base_model": model_name,
        "labels": LABELS,
        "id2label": {str(i): label for label, i in LABELS.items()},
        "label2id": LABELS,
        "version": next_version,
        "adapter_based": True  # Add flag to indicate this is adapter-based
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
    
    # Create a version_info.json file at the standard location to track current version
    with open(os.path.join(models_base_dir, "version_info.json"), "w") as f:
        json.dump({
            "current_version": next_version,
            "latest_model_path": versioned_output_dir,
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    
    print("Training completed successfully!")
    return versioned_output_dir

def load_adapter_model(model_path):
    """Load a model with adapters for inference"""
    
    # Load the model configuration
    with open(os.path.join(model_path, "model_config.json"), "r") as f:
        model_config = json.load(f)
    
    base_model = model_config.get("base_model", MODEL_NAME)
    adapter_name = model_config.get("adapter_name", "ner_adapter")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load base model
    model = AutoAdapterModel.from_pretrained(base_model)
    
    # Load adapter from saved path
    adapter_path = os.path.join(model_path, "adapters")
    model.load_adapter(adapter_path)
    model.load_head(adapter_path)
    
    # Activate the adapter for inference
    model.set_active_adapters(adapter_name)
    
    return model, tokenizer

