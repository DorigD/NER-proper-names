import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_from_disk
from collections import Counter

# Define paths
MODEL_PATH = "models/roberta-finetuned-ner"
DATASET_PATH = "data/tokenized_train"
ERROR_LOG_PATH = "results/error_analysis.txt"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
dataset = load_from_disk(DATASET_PATH)

LABELS = ["O", "B-PERSON", "I-PERSON"]  # Named entity labels

def align_predictions(predictions, label_ids):
    """
    Convert model predictions (logits) into human-readable NER labels.
    """
    preds = np.argmax(predictions, axis=2)
    true_labels = [[LABELS[l] for l in label_id] for label_id in label_ids]
    pred_labels = [[LABELS[p] for p in pred] for pred in preds]
    return pred_labels, true_labels

def analyze_errors():
    """
    Analyze incorrect predictions to understand common model errors.
    """
    model.eval()
    errors = []

    for example in dataset["test"]:
        inputs = {key: torch.tensor(val).unsqueeze(0) for key, val in example.items() if key in ["input_ids", "attention_mask"]}
        labels = example["labels"]

        with torch.no_grad():
            outputs = model(**inputs).logits

        pred_labels, true_labels = align_predictions(outputs.numpy(), [labels])

        for word, true, pred in zip(tokenizer.convert_ids_to_tokens(example["input_ids"]), true_labels[0], pred_labels[0]):
            if true != pred:  # Log mismatched predictions
                errors.append((word, true, pred))

    # Count error types
    error_counts = Counter([(true, pred) for _, true, pred in errors])

    # Save error log
    with open(ERROR_LOG_PATH, "w") as f:
        f.write("Common Errors in NER Model:\n")
        for (true, pred), count in error_counts.items():
            f.write(f"True: {true} -> Predicted: {pred} | Count: {count}\n")
        f.write("\nDetailed Errors:\n")
        for word, true, pred in errors:
            f.write(f"Word: {word} | True: {true} | Predicted: {pred}\n")

    print(f"Error analysis complete! Results saved in {ERROR_LOG_PATH}")

# Run error analysis
#analyze_errors()
