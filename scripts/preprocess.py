import json
from transformers import AutoTokenizer
from datasets import Dataset
import torch
import os

# Load RoBERTa tokenizer with add_prefix_space=True
MODEL_NAME = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)

# Define the labels (only keep PERSON)
LABELS = {"O": 0, "B-PERSON": 1, "I-PERSON": 2}  # Outside, Beginning, Inside

def load_dataset(json_path):
    """Load dataset from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    # Filter out sentences without any PERSON entities
    filtered_data = []
    total_sentences = len(data)
    
    for sentence in data:
        # Check if at least one tag is a PERSON entity (1 or 2)
        if any(tag in [LABELS["B-PERSON"], LABELS["I-PERSON"]] for tag in sentence["ner_tags"]):
            filtered_data.append(sentence)
    
    discarded_count = total_sentences - len(filtered_data)
    print(f"Discarded {discarded_count} sentences without PERSON entities ({discarded_count/total_sentences*100:.1f}%)")
    print(f"Kept {len(filtered_data)} sentences with PERSON entities")
    
    return filtered_data

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
                aligned_labels.append(-100)  # Ignore padding tokens
            elif word_id != prev_word_id:
                aligned_labels.append(label[word_id])  # First subword gets the label
            else:
                # Subsequent subwords of B-PERSON should be I-PERSON
                if label[prev_word_id] == LABELS["B-PERSON"]:
                    aligned_labels.append(LABELS["I-PERSON"])
                # Subsequent subwords of I-PERSON remain I-PERSON
                else:
                    aligned_labels.append(label[prev_word_id])
            prev_word_id = word_id
        labels.append(aligned_labels)
        print(f"Original Tokens: {examples['tokens'][i]}")
        print(f"Original Labels: {label}")
        print(f"Aligned Labels: {aligned_labels}")

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def preprocess(json_path, save_path):
    """Preprocess dataset and save the tokenized version with train/test splits."""
    data = load_dataset(json_path)
    
    # Print sample of filtered data
    if data:
        print(json.dumps(data[:3], indent=4))  # Print the first 3 examples

    # Create dataset and split into train/test
    dataset = Dataset.from_list(data)
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    # Tokenize both splits
    tokenized_dataset = split_dataset.map(tokenize_and_align_labels, batched=True)
    
    # Save the split and tokenized dataset
    tokenized_dataset.save_to_disk(save_path)
    print(f"Preprocessed dataset with train/test splits saved to {save_path}")

# Get absolute paths for better reliability
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
input_json = os.path.join(project_dir, "data", "json", "conllpp_train.json") 
output_dir = os.path.join(project_dir, "data", "tokenized_train")

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(output_dir), exist_ok=True)

preprocess(input_json, output_dir)