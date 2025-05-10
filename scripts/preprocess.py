import json
import shutil  # Add this import
from transformers import AutoTokenizer
from datasets import Dataset
import os
from utils.config import DATA_DIR, LABELS, AVOIDED_symbols

# Load RoBERTa tokenizer with add_prefix_space=True
MODEL_NAME = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
def split_text_into_manageable_chunks(data, max_tokens=100):
    """Split text using punctuation when possible, fallback to fixed chunks"""
    new_data = []
    
    # Common sentence ending punctuation
    sentence_endings = [".", "!", "?", ";", ":", "\n"]
    
    for example in data:
        tokens = example["tokens"]
        ner_tags = example["ner_tags"]
        
        # Skip if already smaller than max size
        if len(tokens) <= max_tokens:
            new_data.append(example)
            continue
            
        # Find potential break points
        break_points = [0]  # Start with beginning
        
        for i, token in enumerate(tokens):
            if token in sentence_endings:
                break_points.append(i + 1)
                
        # Always include end
        if break_points[-1] != len(tokens):
            break_points.append(len(tokens))
            
        # Create chunks between break points, further splitting if needed
        for i in range(len(break_points) - 1):
            start = break_points[i]
            end = break_points[i + 1]
            
            # If chunk is too big, split further
            if end - start > max_tokens:
                # Split into smaller chunks
                for j in range(start, end, max_tokens // 2):
                    sub_end = min(j + max_tokens, end)
                    new_data.append({
                        "tokens": tokens[j:sub_end],
                        "ner_tags": ner_tags[j:sub_end]
                    })
            else:
                # Use the natural break
                new_data.append({
                    "tokens": tokens[start:end],
                    "ner_tags": ner_tags[start:end]
                })
    
    return new_data

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
                # Check if the token is a standalone avoided symbol
                token = examples["tokens"][i][word_id]
                # Ensure we're comparing with a string, not a list
                if isinstance(token, str) and token in AVOIDED_symbols:
                    aligned_labels.append(LABELS["O"])  # Force "O" label for avoided symbols
                else:
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

def preprocess(json_path=os.path.join(DATA_DIR, "json", "result.json"), 
               save_path=os.path.join(DATA_DIR, "tokenized_train"),
               is_testing=False,
               ds_name="default"):
    """
    Preprocess dataset and save the tokenized version with train/validation/test splits.
    Also saves just the training split to a separate directory for accumulating training data.
    
    Args:
        json_path: Path to JSON file with NER data
        save_path: Path where to save the tokenized dataset
        is_testing: If True, all data will be placed in the test split for evaluation
        ds_name: Name to use when saving just the training split (default: "default")
    """
    data = load_dataset(json_path)
    
    # Create dataset from the data
    dataset = Dataset.from_list(data)
    data = split_text_into_manageable_chunks(data)
    # Create a DatasetDict with appropriate splits
    from datasets import DatasetDict
    
    if is_testing:
        # For testing/evaluation: put all data in the test split
        split_dataset = DatasetDict({
            "train": dataset.select(range(0, 0)),  # Empty dataset
            "validation": dataset.select(range(0, 0)),  # Empty dataset
            "test": dataset  # All data goes to test
        })
        print(f"Testing mode: All {len(dataset)} examples placed in the test split")
    else:
        # For standard training: split into train/validation/test
        # First create train and temp splits (80/20)
        train_temp_split = dataset.train_test_split(test_size=0.2, seed=42)
        
        # Then split temp into validation and test (50/50, or 10/10 of original)
        val_test_split = train_temp_split["test"].train_test_split(test_size=0.5, seed=42)
        
        # Create dataset dictionary
        split_dataset = DatasetDict({
            "train": train_temp_split["train"],
            "validation": val_test_split["train"],
            "test": val_test_split["test"]
        })
        
        # Check for person entities in each split and print statistics
        for split_name, split_data in split_dataset.items():
            # Count person entities in this split
            person_count = 0
            total_examples = len(split_data)
            
            for example in split_data:
                contains_person = any(tag in [LABELS["B-PERSON"], LABELS["I-PERSON"]] 
                                    for tag in example["ner_tags"])
                if contains_person:
                    person_count += 1
                    
            print(f"{split_name} split: {person_count}/{total_examples} examples with person entities ({person_count/total_examples*100:.1f}%)")
            
            # Warning if no person entities in test or validation
            if person_count == 0 and split_name in ["test", "validation"]:
                print(f"WARNING: No person entities in {split_name} split! Consider re-running with a different seed.")
    
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
    print(f"Preprocessed dataset saved to {save_path}")
    
    # Save just the train split to ds folder with the provided name
    ds_path = os.path.join(DATA_DIR, "ds", f"{ds_name}")
    os.makedirs(os.path.join(DATA_DIR, "ds"), exist_ok=True)
    
    # Create a dataset with only the train split
    train_only_dataset = DatasetDict({"train": tokenized_dataset["train"]})
    
    # Check if dataset with this name already exists
    if os.path.exists(ds_path):
        try:
            # Load existing dataset
            existing_dataset = DatasetDict.load_from_disk(ds_path)
            # Combine existing train data with new train data
            combined_train = Dataset.concatenate([existing_dataset["train"], train_only_dataset["train"]])
            # Create new dataset with combined data
            train_only_dataset = DatasetDict({"train": combined_train})
            print(f"Added {len(tokenized_dataset['train'])} examples to existing dataset '{ds_name}'")
        except Exception as e:
            print(f"Warning: Could not append to existing dataset: {e}")
    
    # Save the train-only dataset
    train_only_dataset.save_to_disk(ds_path)
    print(f"Training split saved to {ds_path}")