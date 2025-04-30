import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from .metrics import extract_entities
from .config import PROJECT_DIR

def analyze_errors(id2label, all_labels, all_predictions, test_dataset, tokenizer, visualize=False):
    """Analyze prediction errors"""
    # Find where predictions don't match labels
    error_indices = [i for i, (true, pred) in enumerate(zip(all_labels, all_predictions)) if true != pred]
    
    # Only consider errors on actual named entities (not O tags)
    person_error_indices = [i for i in error_indices if all_labels[i] in [1, 2] or all_predictions[i] in [1, 2]]
    
    # Categorize errors by type
    error_types = {}
    for i in person_error_indices:
        true_label = id2label[str(all_labels[i])]
        pred_label = id2label[str(all_predictions[i])]
        error_key = f"{true_label} → {pred_label}"
        
        if error_key not in error_types:
            error_types[error_key] = 0
        error_types[error_key] += 1
    
    if visualize:
        # Display error distribution
        print("\nPERSON Entity Error Distribution:")
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"{error_type}: {count} occurrences")
    
    # Entity boundary analysis - find partial matches
    true_entities = extract_entities(all_labels)
    pred_entities = extract_entities(all_predictions)
    
    # Filter for PERSON entities only
    true_person_entities = [(start, end, type) for start, end, type in true_entities if type == "PERSON"]
    pred_person_entities = [(start, end, type) for start, end, type in pred_entities if type == "PERSON"]
    
    # Find entities that partially overlap but aren't exact matches
    partial_matches = []
    missed_entities = []
    
    for true_entity in true_person_entities:
        true_start, true_end, true_type = true_entity
        found_exact = any(true_entity == pred_entity for pred_entity in pred_person_entities)
        
        if found_exact:
            continue
            
        # Look for partial matches
        found_partial = False
        for pred_entity in pred_person_entities:
            pred_start, pred_end, pred_type = pred_entity
            
            # Check for partial overlap
            if (max(true_start, pred_start) <= min(true_end, pred_end)):
                partial_matches.append((true_entity, pred_entity))
                found_partial = True
                break
        
        if not found_partial:
            missed_entities.append(true_entity)
    
    if visualize:
        print(f"\nPartial PERSON entity matches (boundary errors): {len(partial_matches)}")
        if partial_matches:
            print("Examples of boundary errors:")
            for true_entity, pred_entity in partial_matches[:5]:
                print(f"True: {true_entity}, Predicted: {pred_entity}")
        
        print(f"\nCompletely missed PERSON entities: {len(missed_entities)}")
        if missed_entities:
            print("Examples of missed entities:")
            for entity in missed_entities[:5]:
                print(f"{entity}")
    
    return {
        "error_types": {k: v for k, v in sorted(error_types.items(), key=lambda x: x[1], reverse=True)}, 
        "partial_matches": len(partial_matches), 
        "missed_entities": len(missed_entities)
    }

def visualize_entity_performance(true_entities, pred_entities):
    """Visualize entity performance by length"""
    results_dir = os.path.join(PROJECT_DIR, "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Filter out TITLE entities
    true_entities_filtered = [(start, end, type) for start, end, type in true_entities if type == "PERSON"]
    pred_entities_filtered = [(start, end, type) for start, end, type in pred_entities if type == "PERSON"]
    
    # Entity length analysis
    true_lengths = [end - start + 1 for start, end, _ in true_entities_filtered]
    pred_lengths = [end - start + 1 for start, end, _ in pred_entities_filtered]
    
    # Plotting entity length distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Distribution of entity lengths
    bins = range(1, max(max(true_lengths, default=1), max(pred_lengths, default=1)) + 2)
    ax1.hist(true_lengths, bins=bins, alpha=0.7, label='True Entities')
    ax1.hist(pred_lengths, bins=bins, alpha=0.7, label='Predicted Entities')
    ax1.set_xlabel('Entity Length (tokens)')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of PERSON Entity Lengths')
    ax1.legend()
    
    # Performance by entity length
    success_by_length = {}
    
    for true_entity in true_entities_filtered:
        start, end, _ = true_entity
        length = end - start + 1
        
        if length not in success_by_length:
            success_by_length[length] = {"total": 0, "correct": 0}
        
        success_by_length[length]["total"] += 1
        if true_entity in pred_entities_filtered:
            success_by_length[length]["correct"] += 1
    
    lengths = sorted(success_by_length.keys())
    if lengths:  # Check if there are any lengths to plot
        accuracy = [success_by_length[l]["correct"] / success_by_length[l]["total"] for l in lengths]
        
        ax2.bar(lengths, accuracy)
        ax2.set_xlabel('Entity Length (tokens)')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Recognition Accuracy by Entity Length')
        ax2.set_ylim(0, 1.1)
        
        for i, v in enumerate(accuracy):
            ax2.text(lengths[i], v + 0.02, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'entity_length_analysis.png'))
    plt.show()

def display_context_examples(tokenizer, test_dataset, all_labels, all_predictions, num_examples=5):
    """Display context examples of errors"""
    # Extract entities
    true_entities = extract_entities(all_labels)
    pred_entities = extract_entities(all_predictions)
    
    # Filter to only PERSON entities
    true_person = [(s, e, t) for s, e, t in true_entities if t == "PERSON"]
    pred_person = [(s, e, t) for s, e, t in pred_entities if t == "PERSON"]
    
    # Find misclassified entities
    misclassified = set(true_person) - set(pred_person)
    
    print(f"\nExamples of misclassified PERSON entities ({len(misclassified)} total):")
    
    # Track position in the tokenized test dataset
    token_idx = 0
    examples_shown = 0
    batch_map = {}  # Maps token positions to dataset batch indexes
    
    # Build mapping of token positions to examples
    for i, batch in enumerate(test_dataset):
        labels = batch["labels"]
        valid_tokens = sum(1 for l in labels if l != -100)
        batch_map[i] = (token_idx, token_idx + valid_tokens)
        token_idx += valid_tokens
    
    # Show examples
    shown_errors = set()
    
    for error_entity in sorted(misclassified)[:num_examples*2]:  # Get more than needed in case we can't find some
        if examples_shown >= num_examples:
            break
            
        start_pos, end_pos, _ = error_entity
        
        # Find which example this entity belongs to
        for batch_idx, (batch_start, batch_end) in batch_map.items():
            if batch_start <= start_pos < batch_end:
                # Found the right example
                if error_entity in shown_errors:
                    continue
                    
                shown_errors.add(error_entity)
                
                # Get example text
                example = test_dataset[batch_idx]
                tokens = tokenizer.convert_ids_to_tokens(example['input_ids'])
                
                # Adjust positions to be relative to this example
                rel_start = start_pos - batch_start
                rel_end = end_pos - batch_start
                
                # Find what the model predicted for this span
                local_span = set()
                for pred_start, pred_end, pred_type in pred_person:
                    if max(pred_start, start_pos) <= min(pred_end, end_pos):
                        local_span.add((pred_start, pred_end, pred_type))
                
                # Show the context with highlighting
                context_start = max(0, rel_start - 10)
                context_end = min(len(tokens), rel_end + 10)
                
                context = tokenizer.convert_tokens_to_string(tokens[context_start:context_end])
                entity_text = tokenizer.convert_tokens_to_string(tokens[rel_start:rel_end+1])
                
                print(f"\nExample {examples_shown+1}:")
                print(f"Context: ...{context}...")
                print(f"True entity: {entity_text} [{start_pos}:{end_pos}] (PERSON)")
                if local_span:
                    for pred in local_span:
                        pred_text = f"[{pred[0]}:{pred[1]}] ({pred[2]})"
                        print(f"Partial match: {pred_text}")
                else:
                    print("Predicted: Not detected as entity")
                
                examples_shown += 1
                break

def analyze_confidence(model, device, test_dataset, all_labels, all_predictions):
    """Analyze model confidence"""
    results_dir = os.path.join(PROJECT_DIR, "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)
    
    all_confidences = []
    all_correctness = []
    
    token_idx = 0
    
    for batch in test_dataset:
        input_ids = torch.tensor([batch["input_ids"]]).to(device)
        attention_mask = torch.tensor([batch["attention_mask"]]).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get softmax probabilities
        logits = outputs.logits.cpu().numpy()[0]
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        
        # Get confidence of selected class
        predictions = np.argmax(logits, axis=1)
        confidences = [probs[i, p] for i, p in enumerate(predictions)]
        
        # Match with labels
        labels = batch["labels"]
        
        # Process non-padding tokens
        for i, (conf, pred, label) in enumerate(zip(confidences, predictions, labels)):
            if label == -100:
                continue
                
            is_correct = (pred == label)
            
            all_confidences.append(conf)
            all_correctness.append(is_correct)
            token_idx += 1
    
    # Plot confidence distributions
    correct_conf = [c for c, correct in zip(all_confidences, all_correctness) if correct]
    wrong_conf = [c for c, correct in zip(all_confidences, all_correctness) if not correct]
    
    plt.figure(figsize=(12, 6))
    plt.hist(correct_conf, bins=20, alpha=0.5, label='Correct Predictions')
    plt.hist(wrong_conf, bins=20, alpha=0.5, label='Incorrect Predictions')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.title('Confidence Distribution: Correct vs Incorrect Predictions')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confidence_analysis.png'))
    plt.show()
    
    # Calculate metrics by confidence threshold
    thresholds = np.arange(0.5, 1.0, 0.05)
    results = []
    
    for threshold in thresholds:
        # Count predictions above threshold
        above_threshold = [correct for conf, correct in zip(all_confidences, all_correctness) 
                          if conf >= threshold]
        
        if above_threshold:
            accuracy = sum(above_threshold) / len(above_threshold)
            coverage = len(above_threshold) / len(all_confidences)
            results.append((threshold, accuracy, coverage))
    
    # Plot threshold analysis
    if results:  # Check if we have results to plot
        thresholds, accuracies, coverages = zip(*results)
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        ax1.plot(thresholds, accuracies, 'b-', label='Accuracy')
        ax1.set_xlabel('Confidence Threshold')
        ax1.set_ylabel('Accuracy', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        ax2 = ax1.twinx()
        ax2.plot(thresholds, coverages, 'r-', label='Coverage')
        ax2.set_ylabel('Coverage', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        fig.tight_layout()
        plt.title('Accuracy vs Coverage by Confidence Threshold')
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        plt.savefig(os.path.join(results_dir, 'threshold_analysis.png'))
        plt.show()
    
    return {"mean_confidence_correct": np.mean(correct_conf), 
            "mean_confidence_incorrect": np.mean(wrong_conf) if wrong_conf else 0}
    
# Add this function before the evaluate method or at the top of the file

# Replace the existing convert_numpy_to_python_types function with this more robust version

def convert_numpy_to_python_types(obj):
    """Convert NumPy types to Python native types for JSON serialization"""
    import numpy as np
    
    if isinstance(obj, dict):
        return {key: convert_numpy_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python_types(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return convert_numpy_to_python_types(obj.tolist())
    elif isinstance(obj, np.number):  # This will catch all NumPy numeric types
        return obj.item()  # Convert to native Python type
    else:
        return obj