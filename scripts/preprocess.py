import json
from transformers import AutoTokenizer
from datasets import Dataset
import torch
import os
from utils.config import DATA_DIR

# Define the label mapping
LABELS = {"O": 0, "B-PERSON": 1, "I-PERSON": 2, "TITLE": 3}

# Load RoBERTa tokenizer with add_prefix_space=True
MODEL_NAME = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)


def load_dataset(json_path):
    """Load dataset from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def tokenize_and_align_labels(examples):
    """Tokenize text and align NER labels with subword tokens."""
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to words
        aligned_labels = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)  # Ignore special tokens (CLS, SEP, etc.)
            elif word_id != prev_word_id:
                aligned_labels.append(label[word_id])  # First subword gets the label
            else:
                # For subsequent subwords of a token:
                if label[prev_word_id] == LABELS["B-PERSON"]:
                    aligned_labels.append(LABELS["I-PERSON"])  # B-PERSON -> I-PERSON
                else:
                    aligned_labels.append(label[prev_word_id])  # Keep the same label
            prev_word_id = word_id
            
        labels.append(aligned_labels)
        


    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def preprocess(json_path=os.path.join(DATA_DIR, "json", "result.json"), 
               save_path=os.path.join(DATA_DIR, "tokenized_train")):
    """Preprocess dataset and save the tokenized version with train/test splits."""
    data = load_dataset(json_path)
    print(f"Loaded {len(data)} examples from {json_path}")
    
    if len(data) > 0:

        # Inspect a sample to determine the tag format
        sample_tags = data[0]["ner_tags"] if data else []

        
        # Verify the data format matches our expectations
        expected_labels = set(LABELS.values())
        all_labels = set()
        for entry in data[:100]:  # Check first 100 entries
            all_labels.update(entry["ner_tags"])
        

    
    # Create dataset and split into train/test
    dataset = Dataset.from_list(data)
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    tokenized_dataset = split_dataset.map(
        tokenize_and_align_labels, 
        batched=True,
        batch_size=32,
        remove_columns=["tokens", "ner_tags"]  # Remove the original columns
    )
    
    # Ensure the output directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Save the split and tokenized dataset
   
    tokenized_dataset.save_to_disk(save_path)
   
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Training examples: {len(tokenized_dataset['train'])}")
    print(f"Testing examples: {len(tokenized_dataset['test'])}")
    print(f"Features: {list(tokenized_dataset['train'].features.keys())}")


# Get absolute paths for better reliability
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
input_json = os.path.join(project_dir, "data", "json", "result.json")
output_dir = os.path.join(project_dir, "data", "tokenized_train")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Run preprocessing
preprocess(input_json, output_dir) 