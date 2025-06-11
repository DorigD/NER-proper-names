import os
import json
import random
import sys
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from tqdm import tqdm
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import DATA_DIR
from datetime import datetime
PROCESSED_DATA_DIR = DATA_DIR


# Default mapping definitions
INPUT_TAGS = {"O": 0, "B-PERSON": 1, "I-PERSON": 2, "TITLE": 3}
# Two possible output formats:
OUTPUT_TAGS_WITH_TITLE = {"O": 0, "B-PERSON": 1, "I-PERSON": 2, "B-TITLE": 3, "I-TITLE": 4}
OUTPUT_TAGS_PERSON_ONLY = {"O": 0, "B-PERSON": 1, "I-PERSON": 2}


def convert_tags_to_bio(
    tags: List[int], 
    tokens: List[str], 
    input_mapping: Dict[str, int] = INPUT_TAGS,
    output_mapping: Dict[str, int] = OUTPUT_TAGS_WITH_TITLE
) -> List[int]:
    """
    Convert tags from input format to BIO format
    
    Args:
        tags: List of tag IDs
        tokens: List of corresponding tokens
        input_mapping: Dictionary mapping tag names to IDs in input data
        output_mapping: Dictionary mapping tag names to IDs in output BIO format
    
    Returns:
        List of tag IDs in BIO format
    """
    # Create reverse mapping from ID to tag name
    rev_input_map = {v: k for k, v in input_mapping.items()}
    
    # Initialize output tags
    bio_tags = []
    
    for i, tag_id in enumerate(tags):
        # Get tag name from input mapping
        tag_name = rev_input_map.get(tag_id, "O")
        
        # Handle already BIO format tags (O, B-PERSON, I-PERSON)
        if tag_name in output_mapping:
            bio_tags.append(output_mapping[tag_name])
        # Handle TITLE tag (convert to B-TITLE or I-TITLE)
        elif tag_name == "TITLE":
            # Check if this is the beginning of a TITLE entity
            if i == 0 or rev_input_map.get(tags[i-1], "O") != "TITLE":
                # First token of TITLE entity
                if "B-TITLE" in output_mapping:
                    bio_tags.append(output_mapping["B-TITLE"])
                else:
                    # If we're not using TITLE tags in output, map to O
                    bio_tags.append(output_mapping["O"])
            else:
                # Continuation of TITLE entity
                if "I-TITLE" in output_mapping:
                    bio_tags.append(output_mapping["I-TITLE"])
                else:
                    # If we're not using TITLE tags in output, map to O
                    bio_tags.append(output_mapping["O"])
        # Default to O for unknown tags
        else:
            bio_tags.append(output_mapping["O"])
            
    return bio_tags


def load_json_data(file_path: str) -> List[Dict]:
    """
    Load NER data from JSON file or JSON lines file
    """
    data = []
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
        
    # Determine file format and load accordingly
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            # Try loading as a JSON array
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError:
                # Try loading as JSONL (one JSON object per line)
                f.seek(0)
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
    elif file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
        
    return data


def create_dataset_splits(
    data: List[Dict], 
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42
) -> Dict[str, List[Dict]]:
    """
    Split data into train, validation, and test sets
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # First split: train+val / test
    train_val, test = train_test_split(data, test_size=test_size, random_state=seed)
    
    # Second split: train / val
    if val_size > 0:
        # Adjust validation size to account for data already allocated to test
        adjusted_val_size = val_size / (1 - test_size)
        train, val = train_test_split(train_val, test_size=adjusted_val_size, random_state=seed)
    else:
        train, val = train_val, []
        
    return {
        "train": train,
        "validation": val,
        "test": test
    }


def preprocess_for_token_classification(
    examples: Dict,
    tokenizer,
    max_length: int = 128,
    label_all_tokens: bool = True
) -> Dict:
    """
    Tokenize and align labels for token classification tasks
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
        padding="max_length"
    )
    
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            # Special tokens get label -100
            if word_idx is None:
                label_ids.append(-100)
            # For regular tokens:
            elif word_idx != previous_word_idx:
                # First token of a word gets the label
                label_ids.append(label[word_idx])
            else:
                # For other tokens of a word:
                if label_all_tokens:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
            previous_word_idx = word_idx
            
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def create_ner_dataset(
    file_path: str = os.path.join(DATA_DIR, "json", "results.json"),
    model_name: str = "roberta-base",
    output_dir: str = None,  # If None, will save to tokenized_train
    include_title_tags: bool = True,
    val_size: float = 0.1,
    test_size: float = 0.1,
    max_length: int = 128,
    seed: int = 42,
    save_test_separately: bool = True,  # Save test split in ds directory
    test_output_dir: str = None  # Directory for test split in ds structure
) -> DatasetDict:
    """
    Create an NER dataset from a JSON file with proper BIO tags
    
    Args:
        file_path: Path to the input JSON file
        model_name: Name of the pretrained model to use for tokenization
        output_dir: Directory path to save the full dataset (if None, saves to tokenized_train)
        include_title_tags: Whether to include TITLE tags in the output
        val_size: Fraction of data to use for validation
        test_size: Fraction of data to use for testing
        max_length: Maximum sequence length for tokenization
        seed: Random seed for dataset splitting
        save_test_separately: Whether to save test split in ds directory structure
        test_output_dir: Full path for test split (if None, auto-generated in ds structure)
        
    Returns:
        HuggingFace DatasetDict with train, validation, and test splits
    """
    # Load data
    print(f"Loading data from {file_path}")
    data = load_json_data(file_path)
    
    # Determine output tag mapping
    output_mapping = OUTPUT_TAGS_WITH_TITLE if include_title_tags else OUTPUT_TAGS_PERSON_ONLY
    print(f"Using output tag mapping: {output_mapping}")
    
    # Convert tags to BIO format
    print("Converting tags to BIO format")
    for item in tqdm(data):
        item["labels"] = convert_tags_to_bio(
            item["ner_tags"], 
            item["tokens"],
            INPUT_TAGS, 
            output_mapping
        )
        
    # Split data
    print(f"Splitting data (val={val_size}, test={test_size})")
    splits = create_dataset_splits(data, val_size, test_size, seed)
    
    # Create HuggingFace datasets
    datasets = DatasetDict({
        split_name: Dataset.from_dict({
            "tokens": [item["tokens"] for item in split_data],
            "labels": [item["labels"] for item in split_data]
        })
        for split_name, split_data in splits.items()
    })
    
    # Load tokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    
    # Tokenize and align labels
    print("Tokenizing and aligning labels")
    tokenized_datasets = DatasetDict({
        split_name: dataset.map(
            lambda examples: preprocess_for_token_classification(
                examples, tokenizer, max_length
            ),
            batched=True,
            remove_columns=["tokens", "labels"]
        )
        for split_name, dataset in datasets.items()
    })
    
    # ALWAYS save the full dataset with all 3 splits to tokenized_train directory
    if output_dir is None:
        # Default location for full dataset
        main_output_path = os.path.join(DATA_DIR, "tokenized_train")
    else:
        main_output_path = output_dir
        
    print(f"Saving full dataset (train/validation/test) to {main_output_path}")
    os.makedirs(main_output_path, exist_ok=True)
    tokenized_datasets.save_to_disk(main_output_path)
    
    # ALSO save test split separately in ds directory structure for inference
    if save_test_separately and "test" in tokenized_datasets:
        # Generate test output directory path if not provided
        if test_output_dir is None:
            # Create path in ds structure: data/ds/{TITLE|NO-TITLE}/{dataset_name}
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            title_suffix = "TITLE" if include_title_tags else "NO-TITLE"
            test_output_dir = os.path.join(DATA_DIR, "ds", title_suffix, base_name)
        
        print(f"Saving dataset copy for inference to {test_output_dir}")
        os.makedirs(test_output_dir, exist_ok=True)
        
        # Save the FULL dataset structure (train/validation/test) in ds directory too
        # This allows for both training and inference from the same location
        tokenized_datasets.save_to_disk(test_output_dir)
        
        # Also save metadata about the dataset
        dataset_metadata = {
            "source_file": file_path,
            "model_name": model_name,
            "max_length": max_length,
            "include_title_tags": include_title_tags,
            "test_size": test_size,
            "val_size": val_size,
            "seed": seed,
            "num_train_examples": len(tokenized_datasets["train"]),
            "num_val_examples": len(tokenized_datasets["validation"]),
            "num_test_examples": len(tokenized_datasets["test"]),
            "output_mapping": output_mapping,
            "created_at": str(datetime.now())
        }
        
        metadata_path = os.path.join(test_output_dir, "dataset_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_metadata, f, indent=2)
        
        # Save raw test data (before tokenization) for easier analysis
        raw_test_path = os.path.join(test_output_dir, "raw_test_data.json")
        with open(raw_test_path, 'w', encoding='utf-8') as f:
            json.dump(splits["test"], f, indent=2)
        
        print(f"Dataset saved with splits:")
        print(f"  • Train: {len(tokenized_datasets['train'])} examples")
        print(f"  • Validation: {len(tokenized_datasets['validation'])} examples") 
        print(f"  • Test: {len(tokenized_datasets['test'])} examples")
        print(f"Metadata saved to: {metadata_path}")
        print(f"Raw test data saved to: {raw_test_path}")
    
    # Print dataset statistics
    print("\nDataset statistics:")
    for split_name, dataset in tokenized_datasets.items():
        print(f"  {split_name}: {len(dataset)} examples")
    
    return tokenized_datasets
'''
if __name__ == "__main__":
    # Example usage
    create_ner_dataset(
        file_path=os.path.join(DATA_DIR, "json", "conllpp_train.json"),
        model_name="roberta-base",
        output_dir="tokenized_train",
        include_title_tags=True,
        val_size=0.1,
        test_size=0.1,
        max_length=128,
        seed=42
    )
'''