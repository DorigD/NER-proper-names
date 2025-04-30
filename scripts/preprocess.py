import json
import shutil  # Add this import
from transformers import AutoTokenizer
from datasets import Dataset
import os
from utils.config import DATA_DIR

# Load RoBERTa tokenizer with add_prefix_space=True
MODEL_NAME = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)

# Define the labels (only keep PERSON)
LABELS = {"O": 0, "B-PERSON": 1, "I-PERSON": 2}  # Outside, Beginning, Inside

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

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def preprocess(json_path=os.path.join(DATA_DIR, "json", "result.json"), save_path=os.path.join(DATA_DIR, "tokenized_train")):
    """Preprocess dataset and save the tokenized version with train/validation/test splits."""
    data = load_dataset(json_path)
    
    # Inspect a sample to determine the tag format
    sample_tags = data[0]["ner_tags"] if data else []
    
    # Create dataset with three splits: train/validation/test
    dataset = Dataset.from_list(data)
    
    # First create train and temp splits (80/20)
    train_temp_split = dataset.train_test_split(test_size=0.2, seed=42)
    
    # Then split temp into validation and test (50/50, or 10/10 of original)
    val_test_split = train_temp_split["test"].train_test_split(test_size=0.5, seed=42)
    
    # Create a DatasetDict with all three splits
    from datasets import DatasetDict
    split_dataset = DatasetDict({
        "train": train_temp_split["train"],
        "validation": val_test_split["train"],
        "test": val_test_split["test"]
    })
    
    # Tokenize all splits
    tokenized_dataset = split_dataset.map(tokenize_and_align_labels, batched=True)
    
    # Save the split and tokenized dataset
    if os.path.exists(save_path):
        try:
            # Use shutil.rmtree() for directories instead of os.remove()
            shutil.rmtree(save_path)
            print(f"Removed existing directory: {save_path}")
        except Exception as e:
            print(f"Warning: Could not remove existing directory: {e}")
    
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the dataset
    tokenized_dataset.save_to_disk(save_path)
    print(f"Preprocessed dataset with train/validation/test splits saved to {save_path}")

"""
# Get absolute paths for better reliability
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
input_json = os.path.join(project_dir, "data", "json", "conllpp_train.json") 
output_dir = os.path.join(project_dir, "data", "tokenized_train")

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(output_dir), exist_ok=True)

preprocess(input_json, output_dir)
"""