import os
import json
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from datasets import load_from_disk

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
model_output_dir = os.path.join(project_dir, "models", "roberta-finetuned-ner")
dataset_path = os.path.join(project_dir, "data", "tokenized_train")

# Load the model and tokenizer
print("Loading model and tokenizer...")
model = AutoModelForTokenClassification.from_pretrained(model_output_dir)
tokenizer = AutoTokenizer.from_pretrained(model_output_dir)

# Load the tokenized dataset
print("Loading tokenized dataset...")
tokenized_dataset = load_from_disk(dataset_path)

# Function for inference
def infer(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=2)
    predicted_labels = [model.config.id2label[p.item()] for p in predictions[0]]
    return predicted_labels

# Example usage
if __name__ == "__main__":
    sample_text = "Barack Obama was the 44th President of the United States."
    predictions = infer(sample_text)
    print(f"Input text: {sample_text}")
    print(f"Predicted labels: {predictions}")