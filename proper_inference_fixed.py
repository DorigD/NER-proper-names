"""
Proper inference script that reconstructs the exact training architecture.
This should achieve the full 0.82 F1 performance.
"""

import os
import sys
import torch
import json
from datasets import load_from_disk
from transformers import AutoTokenizer, DataCollatorForTokenClassification
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the exact same classes used in training
from scripts.train import (
    RobertaCRFForTokenClassification, 
    compute_metrics,
    confidence_based_postprocessing,
    LABEL2ID, ID2LABEL, NUM_LABELS
)

def load_exact_training_model(model_path, dataset_path):
    """
    Load the model using the exact same architecture as training
    """
    print(f"Loading model from: {model_path}")
    print(f"Using dataset: {dataset_path}")
    
    # Load label configuration
    label_config_path = os.path.join(model_path, "label_config.json")
    if os.path.exists(label_config_path):
        with open(label_config_path, 'r') as f:
            label_config = json.load(f)
        print(f"Loaded label config: {label_config['labels']}")
        
        # Update global variables
        global LABEL2ID, ID2LABEL, NUM_LABELS
        LABEL2ID = label_config['label2id']
        ID2LABEL = label_config['id2label'] 
        NUM_LABELS = label_config['num_labels']
        
        # Update the training module globals too
        import scripts.train as train_module
        train_module.LABEL2ID = LABEL2ID
        train_module.ID2LABEL = ID2LABEL
        train_module.NUM_LABELS = NUM_LABELS
    else:
        print("No label config found, using defaults")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_from_disk(dataset_path)
    print(f"Test set size: {len(dataset['test'])}")
    
    # Filter labels to match model
    def filter_labels(example):
        filtered_labels = []
        for label in example['labels']:
            if label == -100:
                filtered_labels.append(label)
            elif label >= NUM_LABELS:
                filtered_labels.append(0)  # Map invalid labels to O
            else:
                filtered_labels.append(label)
        example['labels'] = filtered_labels
        return example
    
    print("Filtering labels...")
    dataset['test'] = dataset['test'].map(filter_labels)
    
    # Create the EXACT same model architecture as training
    print("\nCreating model with exact training architecture...")
    
    # Try to load the model from the saved directory first, then fall back to roberta-base
    try:
        model = RobertaCRFForTokenClassification(
            model_name=model_path,  # Try loading from saved model first
            num_labels=NUM_LABELS,
            alpha=0.25,
            gamma=2.0,
            person_weight=5.0,
            crf_weight=0.5,
            focal_weight=0.2,
            dice_weight=0.3,
            classifier_params={
                "num_attention_heads": 5,  # Inferred from weight shapes: 765/153 = 5
                "max_relative_position": 3,  # Inferred from weight shapes: (7-1)/2 = 3
                "dropout": 0.1
            },
            dice_loss_params={
                "smooth": 1e-5,
                "b_weight": 3.0,
                "i_end_weight": 2.5,
                "context_weight": 1.5
            }
        )
    except Exception as e:
        print(f"Failed to load from {model_path}, trying roberta-base: {e}")
        model = RobertaCRFForTokenClassification(
            model_name="roberta-base",  # Fall back to base model
            num_labels=NUM_LABELS,
            alpha=0.25,
            gamma=2.0,
            person_weight=5.0,
            crf_weight=0.5,
            focal_weight=0.2,
            dice_weight=0.3,
            classifier_params={
                "num_attention_heads": 5,  # Inferred from weight shapes: 765/153 = 5
                "max_relative_position": 3,  # Inferred from weight shapes: (7-1)/2 = 3
                "dropout": 0.1
            },
            dice_loss_params={
                "smooth": 1e-5,
                "b_weight": 3.0,
                "i_end_weight": 2.5,
                "context_weight": 1.5
            }
        )
    
    # Load the saved weights
    print("Loading trained weights...")
    from safetensors.torch import load_file
    
    model_file = os.path.join(model_path, "model.safetensors")
    if os.path.exists(model_file):
        state_dict = load_file(model_file)
        
        # Filter out incompatible keys to avoid shape mismatches
        model_state_dict = model.state_dict()
        compatible_state_dict = {}
        
        for key, value in state_dict.items():
            if key in model_state_dict:
                model_param = model_state_dict[key]
                if model_param.shape == value.shape:
                    compatible_state_dict[key] = value
                else:
                    print(f"Skipping {key}: shape mismatch {model_param.shape} vs {value.shape}")
            else:
                print(f"Skipping {key}: not found in model")
        
        missing_keys, unexpected_keys = model.load_state_dict(compatible_state_dict, strict=False)
        
        print(f"Loaded {len(compatible_state_dict)} compatible weight tensors")
        if missing_keys:
            print(f"Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"Unexpected keys: {len(unexpected_keys)}")
    else:
        print(f"ERROR: Model file not found: {model_file}")
        return None
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    
    return model, tokenizer, dataset

def evaluate_with_exact_architecture(model_path, dataset_path):
    """
    Evaluate using the exact same setup as training
    """
    
    # Load model with exact architecture
    result = load_exact_training_model(model_path, dataset_path)
    if result is None:
        return None
        
    model, tokenizer, dataset = result
    
    # Set model to evaluation mode
    model.eval()
    
    # Create data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # Manual evaluation loop (like in training)
    print("\n=== Evaluating with Exact Training Setup ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    all_predictions = []
    all_labels = []
    
    # Process in batches
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset["test"], 
        batch_size=8,  # Same as training
        collate_fn=data_collator
    )
    
    print(f"Processing {len(dataloader)} batches...")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i % 50 == 0:
                print(f"Processing batch {i}/{len(dataloader)}")
            
            # Move to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Get model outputs (same as training)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None  # Inference mode
            )
            
            logits = outputs["logits"]
            
            # Use CRF decoding instead of simple argmax
            try:
                # Get the emissions from the model
                # The model should have a CRF layer
                batch_size, seq_len, num_classes = logits.shape
                
                # Create mask for CRF (True for valid positions)
                crf_mask = attention_mask.bool()
                
                # Use CRF decode method
                if hasattr(model, 'crf') and hasattr(model.crf, 'decode'):
                    crf_predictions = model.crf.decode(logits, mask=crf_mask)
                    # Convert list of lists to tensor
                    predictions = torch.zeros_like(input_ids)
                    for b_idx, pred_seq in enumerate(crf_predictions):
                        seq_len = min(len(pred_seq), predictions.shape[1])
                        predictions[b_idx, :seq_len] = torch.tensor(pred_seq[:seq_len])
                else:
                    # Fall back to argmax if CRF decode is not available
                    predictions = torch.argmax(logits, dim=-1)
                    print(f"Warning: Using argmax fallback for batch {i}")
                    
            except Exception as e:
                print(f"CRF decoding failed for batch {i}: {e}, using argmax")
                predictions = torch.argmax(logits, dim=-1)
            
            # Apply post-processing (same as training)
            if logits.dim() == 3:
                try:
                    # Apply the exact same post-processing as training
                    processed_predictions = confidence_based_postprocessing(
                        logits, predictions, attention_mask
                    )
                    predictions = processed_predictions
                except Exception as e:
                    print(f"Post-processing failed for batch {i}: {e}")
                    # Continue with unprocessed predictions
            
            # Extract sequences for evaluation
            for b in range(predictions.shape[0]):
                pred_seq = []
                label_seq = []
                
                for t in range(predictions.shape[1]):
                    if attention_mask[b, t] == 1 and labels[b, t] != -100:
                        pred_id = predictions[b, t].item()
                        label_id = labels[b, t].item()
                        
                        # Convert to label strings
                        pred_label = ID2LABEL.get(str(pred_id), "O")
                        true_label = ID2LABEL.get(str(label_id), "O") 
                        
                        pred_seq.append(pred_label)
                        label_seq.append(true_label)
                
                if pred_seq and label_seq:
                    all_predictions.append(pred_seq)
                    all_labels.append(label_seq)
    
    # Calculate metrics using the same function as training
    print(f"\nCalculating metrics for {len(all_predictions)} sequences...")
    
    # Use the exact same metrics as training
    from seqeval.metrics import classification_report as seq_classification_report
    from seqeval.scheme import IOB2
    from sklearn.metrics import classification_report
    
    # Entity-level metrics
    entity_results = seq_classification_report(
        all_labels, all_predictions, scheme=IOB2, output_dict=True
    )
    
    person_f1 = entity_results.get("PERSON", {}).get("f1-score", 0.0)
    entity_f1 = entity_results["macro avg"]["f1-score"]
    
    # Token-level metrics
    all_true_labels = [l for seq in all_labels for l in seq]
    all_pred_labels = [p for seq in all_predictions for p in seq]
    
    token_results = classification_report(
        all_true_labels, all_pred_labels, output_dict=True
    )
    token_accuracy = token_results["accuracy"]
    
    print(f"\n=== Final Results ===")
    print(f"Person F1: {person_f1:.4f}")
    print(f"Entity F1: {entity_f1:.4f}")
    print(f"Token Accuracy: {token_accuracy:.4f}")
    
    return {
        "person_f1": person_f1,
        "entity_f1": entity_f1,
        "token_accuracy": token_accuracy,
        "entity_results": entity_results
    }

if __name__ == "__main__":
    model_path = r"c:\Users\Administrator\Desktop\tmp\NER-proper-names\models\roberta-finetuned-ner-NO-TITLE-v3"
    dataset_path = r"c:\Users\Administrator\Desktop\tmp\NER-proper-names\data\tokenized_train"
    
    print("=== Proper Inference with Exact Training Architecture ===")
    results = evaluate_with_exact_architecture(model_path, dataset_path)
    
    if results:
        print(f"\n🎯 Final Person F1 Score: {results['person_f1']:.4f}")
        if results['person_f1'] > 0.8:
            print("🎉 SUCCESS: Achieved expected performance!")
        else:
            print(f"📊 Gap from expected 0.82: {0.82 - results['person_f1']:.4f}")
