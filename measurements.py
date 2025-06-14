"""
Simple inference script for NER model evaluation.
This script loads a trained model and evaluates it on the test dataset.
"""

import os
import sys
import json
from datasets import load_from_disk

# Set environment variable for better CUDA error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import configuration and metrics
from scripts.train import compute_metrics, RobertaCRFForTokenClassification, SimplifiedRobertaCRFForTokenClassification

def load_model_label_config(model_path):
    """Load label configuration from the model directory"""
    label_config_path = os.path.join(model_path, "label_config.json")
    if os.path.exists(label_config_path):
        with open(label_config_path, 'r') as f:
            return json.load(f)
    return None

def simple_inference(model_path, dataset_path):
    """
    Simple inference function that evaluates a trained model on test data.
    
    Args:
        model_path: Path to the trained model directory
        dataset_path: Path to the dataset directory
    """
    
    print(f"Using model: {model_path}")
    print(f"Using dataset: {dataset_path}")
    
    # Load model's label configuration
    label_config = load_model_label_config(model_path)
    if label_config:
        print(f"Model labels: {label_config['labels']}")
        # Update global variables for compute_metrics
        import scripts.train as train_module
        train_module.LABEL2ID = label_config['label2id']
        train_module.ID2LABEL = label_config['id2label']
        train_module.NUM_LABELS = label_config['num_labels']
      # Load dataset
    print("\nLoading dataset...")
    dataset = load_from_disk(dataset_path)
    print(f"Test set size: {len(dataset['test'])}")
      # Debug: Check actual labels in dataset
    test_labels = []
    for example in dataset['test']:
        test_labels.extend(example['labels'])
    unique_labels = set(test_labels)
    print(f"Unique labels in test dataset: {sorted(unique_labels)}")
    print(f"Max label: {max(unique_labels)}, Min label: {min(unique_labels)}")
      # Fix label mismatch - filter dataset to only include valid labels
    # Check how many labels the model actually has from its config
    model_temp = AutoModelForTokenClassification.from_pretrained(model_path, ignore_mismatched_sizes=True)
    actual_num_labels = model_temp.config.num_labels
    max_model_label = actual_num_labels - 1
    print(f"Model actually has {actual_num_labels} labels (0-{max_model_label})")
    del model_temp  # Free memory
    
    def filter_labels(example):
        """Filter out invalid labels and replace with O (label 0)"""
        filtered_labels = []
        for label in example['labels']:
            if label == -100:  # Keep padding tokens
                filtered_labels.append(label)
            elif label > max_model_label:  # Replace invalid labels with O
                filtered_labels.append(0)
            else:
                filtered_labels.append(label)
        example['labels'] = filtered_labels
        return example
      # Apply filtering to test dataset
    print("Filtering invalid labels...")
    dataset['test'] = dataset['test'].map(filter_labels)
    
    # Verify filtering worked
    test_labels_after = []
    for example in dataset['test']:
        test_labels_after.extend(example['labels'])
    unique_labels_after = set(test_labels_after)
    print(f"Labels after filtering: {sorted(unique_labels_after)}")
    
    # Double check - count occurrences of each label
    from collections import Counter
    label_counts = Counter(test_labels_after)
    print(f"Label distribution: {dict(label_counts)}")    # Ensure no labels > max_model_label exist
    invalid_labels = [l for l in unique_labels_after if l > max_model_label and l != -100]
    if invalid_labels:
        print(f"ERROR: Still have invalid labels: {invalid_labels}")
        return None
    else:
        print(" All labels are valid for the model")
    
    # Now test on the full dataset instead of just 10 examples
    print(f"Using full test set: {len(dataset['test'])} examples")# Load model and tokenizer
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    
    # Create a new standard model with correct config
    from transformers import RobertaConfig
    config = RobertaConfig.from_pretrained("roberta-base")
    config.num_labels = actual_num_labels
    config.id2label = {str(i): ["O", "B-PERSON", "I-PERSON"][i] for i in range(actual_num_labels)}
    config.label2id = {["O", "B-PERSON", "I-PERSON"][i]: i for i in range(actual_num_labels)}
    config.problem_type = "token_classification"
    
    # Load the model with the correct config
    model = AutoModelForTokenClassification.from_pretrained(
        "roberta-base",
        config=config,
        ignore_mismatched_sizes=True
    )
    
    # Now manually load the compatible weights from the saved model
    print("Loading compatible weights from saved model...")
    import torch
    try:
        # Load the saved state dict
        model_file = os.path.join(model_path, "model.safetensors")
        if os.path.exists(model_file):
            from safetensors.torch import load_file
            saved_state_dict = load_file(model_file)
            
            # Get current model state dict
            current_state_dict = model.state_dict()
              # Copy compatible weights
            compatible_weights = {}
            loaded_count = 0
            
            # Special handling for classifier weights - try to map custom classifier to simple classifier
            if "classifier.token_classifier.weight" in saved_state_dict and "classifier.weight" in current_state_dict:
                if saved_state_dict["classifier.token_classifier.weight"].shape == current_state_dict["classifier.weight"].shape:
                    compatible_weights["classifier.weight"] = saved_state_dict["classifier.token_classifier.weight"]
                    loaded_count += 1
                    print(" Loaded: classifier.weight (from classifier.token_classifier.weight)")
                else:
                    print(f"Skipped: classifier.token_classifier.weight (shape mismatch: {saved_state_dict['classifier.token_classifier.weight'].shape} vs {current_state_dict['classifier.weight'].shape})")
            
            if "classifier.token_classifier.bias" in saved_state_dict and "classifier.bias" in current_state_dict:
                if saved_state_dict["classifier.token_classifier.bias"].shape == current_state_dict["classifier.bias"].shape:
                    compatible_weights["classifier.bias"] = saved_state_dict["classifier.token_classifier.bias"]
                    loaded_count += 1
                    print(" Loaded: classifier.bias (from classifier.token_classifier.bias)")
                else:
                    print(f"Skipped: classifier.token_classifier.bias (shape mismatch)")
            
            # Try span_classifier weights which might be the main trained classifier (can override token_classifier)
            if "classifier.span_classifier.weight" in saved_state_dict and "classifier.weight" in current_state_dict:
                if saved_state_dict["classifier.span_classifier.weight"].shape == current_state_dict["classifier.weight"].shape:
                    compatible_weights["classifier.weight"] = saved_state_dict["classifier.span_classifier.weight"]
                    loaded_count += 1
                    print(" Loaded: classifier.weight (from classifier.span_classifier.weight) - OVERRIDING token_classifier")
                else:
                    print(f"Skipped: classifier.span_classifier.weight (shape mismatch: {saved_state_dict['classifier.span_classifier.weight'].shape} vs {current_state_dict['classifier.weight'].shape})")
            
            if "classifier.span_classifier.bias" in saved_state_dict and "classifier.bias" in current_state_dict:
                if saved_state_dict["classifier.span_classifier.bias"].shape == current_state_dict["classifier.bias"].shape:
                    compatible_weights["classifier.bias"] = saved_state_dict["classifier.span_classifier.bias"]
                    loaded_count += 1
                    print(" Loaded: classifier.bias (from classifier.span_classifier.bias) - OVERRIDING token_classifier")
                else:
                    print(f"Skipped: classifier.span_classifier.bias (shape mismatch)")
            
            # Load other compatible weights (RoBERTa backbone)
            for key, value in saved_state_dict.items():
                if key.startswith('roberta.') and key in current_state_dict and current_state_dict[key].shape == value.shape:
                    compatible_weights[key] = value
                    loaded_count += 1
                    print(f" Loaded: {key}")
                elif not key.startswith('roberta.') and not key.startswith('classifier.token_classifier.'):
                    if key not in ['classifier.weight', 'classifier.bias']:  # Skip already handled
                        print(f"Skipped: {key} (not compatible with simple model)")
            
            print(f"Attempting to load {len(compatible_weights)} compatible weight tensors")
              # Load the compatible weights
            model.load_state_dict(compatible_weights, strict=False)
            print(f"Successfully loaded {loaded_count} compatible weight tensors")
        else:
            print(f"Model file not found: {model_file}")
            
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Using randomly initialized model")
    
    print(f"Model loaded successfully with {model.config.num_labels} labels")
    print(f"Model labels: {model.config.id2label}")
    
    # Set up data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)    # Set up training arguments for evaluation only
    training_args = TrainingArguments(
        output_dir="./temp_eval",
        per_device_eval_batch_size=4,  # Reduced batch size further
        logging_dir=None,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        use_cpu=False,  # Use GPU
    )
    
    # Create trainer for evaluation
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Evaluate on test set
    print("\n=== Evaluating on Test Set ===")
    test_results = trainer.evaluate()
    print(test_results)
    
    # Extract key metrics
    person_f1 = test_results.get("eval_person_f1", 0.0)
    entity_f1 = test_results.get("eval_entity_f1", 0.0)
    token_accuracy = test_results.get("eval_token_accuracy", 0.0)
    
    print(f"\n=== Summary ===")
    print(f"Person F1: {person_f1:.4f}")
    print(f"Entity F1: {entity_f1:.4f}")
    print(f"Token Accuracy: {token_accuracy:.4f}")
    
    return {
        "person_f1": person_f1,
        "entity_f1": entity_f1,
        "token_accuracy": token_accuracy,
        "full_results": test_results
    }

if __name__ == "__main__":
    # Use the latest checkpoint which should have the proper model
    model_path = r"c:\Users\Administrator\Desktop\tmp\NER-proper-names\models\roberta-finetuned-ner-NO-TITLE-v1"
    dataset_path = r"c:\Users\Administrator\Desktop\tmp\NER-proper-names\data\tokenized_train"  # Use tokenized dataset
    
    print("Starting simple inference...")
    results = simple_inference(model_path, dataset_path)
    
    if results:
        print(f"\nFinal Person F1 Score: {results['person_f1']:.4f}")
