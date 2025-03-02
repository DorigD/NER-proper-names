import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_from_disk, load_metric

# Define model and dataset paths
MODEL_PATH = "models/roberta-finetuned-ner"
DATASET_PATH = "data/tokenized_train"
METRIC = load_metric("seqeval")  # For evaluating NER performance

# Load tokenizer, model, and dataset
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
dataset = load_from_disk(DATASET_PATH)

# Extract labels from dataset
LABELS = ["O", "B-PERSON", "I-PERSON"]

def align_predictions(predictions, label_ids):
    """
    Convert model predictions (logits) into human-readable NER labels.
    """
    preds = np.argmax(predictions, axis=2)
    true_labels = [[LABELS[l] for l in label_id] for label_id in label_ids]
    pred_labels = [[LABELS[p] for p in pred] for pred in preds]
    return pred_labels, true_labels

# Run evaluation
model.eval()
all_predictions, all_true_labels = [], []

for example in dataset["test"]:
    inputs = {key: torch.tensor(val).unsqueeze(0) for key, val in example.items() if key in ["input_ids", "attention_mask"]}
    labels = example["labels"]

    with torch.no_grad():
        outputs = model(**inputs).logits

    pred_labels, true_labels = align_predictions(outputs.numpy(), [labels])
    
    all_predictions.extend(pred_labels)
    all_true_labels.extend(true_labels)

# Compute NER metrics
results = METRIC.compute(predictions=all_predictions, references=all_true_labels)
print(results)  # Show precision, recall, and F1-score

# Save results
with open("results/evaluation_results.txt", "w") as f:
    f.write(str(results))

print("Evaluation complete! Results saved in results/evaluation_results.txt")
