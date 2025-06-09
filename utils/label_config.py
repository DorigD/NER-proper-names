"""
Label configuration management for NER models.
This module handles dynamic label schema switching between TITLE and NO-TITLE versions.
"""

import os
import json
from typing import Dict, Any

# Base configurations
LABELS_WITH_TITLE = {"O": 0, "B-PERSON": 1, "I-PERSON": 2, "B-TITLE": 3, "I-TITLE": 4}
LABELS_WITHOUT_TITLE = {"O": 0, "B-PERSON": 1, "I-PERSON": 2}

def get_label_config(include_title: bool = True) -> Dict[str, Any]:
    """
    Get label configuration based on whether to include TITLE tags.
    
    Args:
        include_title: Whether to include TITLE tags in the schema
        
    Returns:
        Dictionary with labels, id2label, label2id, and num_labels
    """
    if include_title:
        labels = LABELS_WITH_TITLE
    else:
        labels = LABELS_WITHOUT_TITLE
    
    return {
        "labels": labels,
        "label2id": labels,
        "id2label": {str(i): label for label, i in labels.items()},
        "num_labels": len(labels)
    }

def save_label_config(output_dir: str, include_title: bool = True):
    """
    Save label configuration to model directory for consistency.
    
    Args:
        output_dir: Directory to save the configuration
        include_title: Whether the model uses TITLE tags
    """
    config = get_label_config(include_title)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(output_dir, "label_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print(f"Label configuration saved to {config_path}")
    return config_path

def load_label_config(model_dir: str) -> Dict[str, Any]:
    """
    Load label configuration from model directory.
    
    Args:
        model_dir: Directory containing the model and configuration
        
    Returns:
        Label configuration dictionary
    """
    config_path = os.path.join(model_dir, "label_config.json")
    
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"Loaded label configuration from {config_path}")
        return config
    else:
        print(f"No label configuration found at {config_path}, using default")
        # Try to infer from model config if available
        model_config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(model_config_path):
            with open(model_config_path, 'r', encoding='utf-8') as f:
                model_config = json.load(f)
            
            num_labels = model_config.get("num_labels", 3)
            if num_labels == 5:
                return get_label_config(include_title=True)
            else:
                return get_label_config(include_title=False)
        
        # Default fallback
        return get_label_config(include_title=False)

def detect_label_schema(dataset_path: str = None, labels: list = None) -> bool:
    """
    Detect whether a dataset uses TITLE tags.
    
    Args:
        dataset_path: Path to the dataset (optional)
        labels: List of labels to check (optional)
        
    Returns:
        True if TITLE tags are present, False otherwise
    """
    if labels:
        # Check if any TITLE-related labels are present
        label_set = set(labels) if isinstance(labels[0], str) else set()
        return any("TITLE" in str(label) for label in label_set)
    
    if dataset_path and os.path.exists(dataset_path):
        try:
            from datasets import load_from_disk
            dataset = load_from_disk(dataset_path)
            
            # Check a sample of labels
            sample_labels = []
            for split in dataset.keys():
                if len(dataset[split]) > 0:
                    sample_labels.extend(dataset[split]["labels"][:10])  # Check first 10 examples
                    break
            
            # Flatten the labels and check for TITLE tags
            flat_labels = [label for seq in sample_labels for label in seq if label >= 0]
            max_label = max(flat_labels) if flat_labels else 0
            
            # If max label > 2, likely has TITLE tags (assuming O=0, B-PERSON=1, I-PERSON=2, B-TITLE=3, I-TITLE=4)
            return max_label > 2
            
        except Exception as e:
            print(f"Error detecting label schema: {e}")
    
    return False
