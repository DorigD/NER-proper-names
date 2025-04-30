from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from transformers import EvalPrediction


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
    
    # Title metrics
    title_metrics = precision_recall_fscore_support(
        [l for sublist in true_labels for l in sublist],
        [p for sublist in true_predictions for p in sublist],
        labels=[3],  # TITLE only
        average="micro"
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
        "person_token_f1": person_metrics[2],  # Changed name for token-level F1
        "b_person_f1": b_person_metrics[2],
        "i_person_f1": i_person_metrics[2],
        "title_f1": title_metrics[2],
        "entity_precision": entity_results["precision"],
        "entity_recall": entity_results["recall"],
        "person_entity_f1": entity_results["f1"],  # Renamed for clarity
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
    entity_type = None
    
    for i, label in enumerate(token_labels):
        if label == 1:  # B-PERSON
            if entity_start is not None:
                entities.append((entity_start, i-1, entity_type))
            entity_start = i
            entity_type = "PERSON"
        elif label == 2:  # I-PERSON
            if entity_start is None:
                # I-PERSON without preceding B-PERSON, treat as B-PERSON
                entity_start = i
                entity_type = "PERSON"
        elif label == 3:  # TITLE
            if entity_start is not None and entity_type != "TITLE":
                # Close previous entity before starting a new one
                entities.append((entity_start, i-1, entity_type))
            if entity_type != "TITLE" or entity_start is None:
                # Start a new TITLE entity
                entity_start = i
                entity_type = "TITLE"
        elif entity_start is not None:
            entities.append((entity_start, i-1, entity_type))
            entity_start = None
            entity_type = None
    
    # Handle entity at the end of sequence
    if entity_start is not None:
        entities.append((entity_start, len(token_labels)-1, entity_type))
        
    return entities
