import torch
import torch.nn as nn
from adapters import AutoAdapterModel

def get_adapter_model_for_token_classification(model_name, num_labels, adapter_name="ner_adapter", 
                                              adapter_config=None):
    """
    Create a standard adapter-based model for token classification without custom layers
    
    Args:
        model_name: Base model name or path
        num_labels: Number of classification labels
        adapter_name: Name to use for the adapter
        adapter_config: Configuration for the adapter
        
    Returns:
        Configured adapter model
    """
    # Load base model with adapter support
    model = AutoAdapterModel.from_pretrained(model_name)
    
    # Add task adapter for NER
    model.add_adapter(adapter_name, config=adapter_config)
    
    # Add classification head
    model.add_classification_head(
        adapter_name,
        num_labels=num_labels
    )
    
    # Activate adapter
    model.train_adapter(adapter_name)
    
    return model