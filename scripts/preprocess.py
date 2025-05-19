import json
import shutil
import random
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

def fix_bio_sequences(data):
    """
    Fix invalid BIO sequences by converting I-PERSON tags to B-PERSON tags 
    when they don't follow a B-PERSON or I-PERSON tag.
    """
    fixed_data = []
    
    for example in data:
        fixed_tags = example["ner_tags"].copy()
        tokens = example["tokens"]
        
        for i in range(len(fixed_tags)):
            # If current tag is I-PERSON
            if fixed_tags[i] == LABELS["I-PERSON"]:
                # Check if it's the first token or previous token was not a PERSON entity
                if i == 0 or (fixed_tags[i-1] != LABELS["B-PERSON"] and fixed_tags[i-1] != LABELS["I-PERSON"]):
                    # Convert to B-PERSON
                    fixed_tags[i] = LABELS["B-PERSON"]
        
        fixed_data.append({
            "tokens": tokens,
            "ner_tags": fixed_tags
        })
    
    return fixed_data

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

def augment_dataset(dataset, augmentation_factor=0.3):
    """
    Augment dataset with entity substitution techniques.
    
    Args:
        dataset: Dataset to augment
        augmentation_factor: Ratio of new examples to generate (relative to original size)
    
    Returns:
        Augmented dataset with original and new examples
    """
    # Extract all person entities from dataset for substitution
    person_entities = {}  # Will store {entity_type: [list of entities]}
    
    print("Collecting entities for augmentation...")
    
    # First pass: collect all entities
    for example in dataset:
        tokens = example["tokens"]
        tags = example["ner_tags"]
        
        i = 0
        while i < len(tags):
            # Check for B- tag (beginning of entity)
            if tags[i] == LABELS["B-PERSON"]:
                # Extract the entity
                entity_tokens = [tokens[i]]
                entity_type = "PERSON"
                j = i + 1
                
                # Continue until end of entity
                while j < len(tags) and tags[j] == LABELS["I-PERSON"]:
                    entity_tokens.append(tokens[j])
                    j += 1
                
                # Store the entity
                if entity_type not in person_entities:
                    person_entities[entity_type] = []
                person_entities[entity_type].append(entity_tokens)
                
                # Move to next token after this entity
                i = j
            else:
                i += 1
    
    # Check if we found any entities to work with
    if not person_entities or "PERSON" not in person_entities or len(person_entities["PERSON"]) <= 1:
        print("Warning: Not enough entities found for augmentation!")
        return dataset
        
    print(f"Found {len(person_entities.get('PERSON', []))} person entities for augmentation")
    
    # Determine number of examples to generate
    num_to_augment = int(len(dataset) * augmentation_factor)
    augmented_examples = []
    
    print(f"Generating {num_to_augment} augmented examples...")
    
    # Generate new examples through entity substitution
    for _ in range(num_to_augment):
        # Select a random example to augment
        example_idx = random.randint(0, len(dataset) - 1)
        example = dataset[example_idx]
        tokens = example["tokens"].copy()
        tags = example["ner_tags"].copy()
        
        # Find entity spans in this example
        entity_spans = []  # Will store (start_idx, end_idx, entity_type)
        i = 0
        while i < len(tags):
            if tags[i] == LABELS["B-PERSON"]:
                start_idx = i
                j = i + 1
                while j < len(tags) and tags[j] == LABELS["I-PERSON"]:
                    j += 1
                entity_spans.append((start_idx, j, "PERSON"))
                i = j
            else:
                i += 1
        
        # Skip if no entities to substitute
        if not entity_spans:
            continue
        
        # Choose a random entity to substitute
        span_idx = random.randint(0, len(entity_spans) - 1)
        start_idx, end_idx, entity_type = entity_spans[span_idx]
        
        # Find a different entity of the same type
        original_entity = tokens[start_idx:end_idx]
        candidates = [e for e in person_entities[entity_type] if e != original_entity]
        
        if not candidates:
            continue
            
        replacement = random.choice(candidates)
        
        # Create new example with substituted entity
        new_tokens = tokens[:start_idx] + replacement + tokens[end_idx:]
        new_tags = tags[:start_idx] + [LABELS["B-PERSON"]] + [LABELS["I-PERSON"]] * (len(replacement) - 1) + tags[end_idx:]
        
        augmented_examples.append({"tokens": new_tokens, "ner_tags": new_tags})
    
    print(f"Created {len(augmented_examples)} new examples through entity substitution")
    
    # Combine original dataset with augmented examples
    all_examples = dataset + augmented_examples
    
    return all_examples

def preprocess(json_path=os.path.join(DATA_DIR, "json", "result.json"), 
               save_path=os.path.join(DATA_DIR, "tokenized_train"),
               is_testing=False,
               ds_name="default",
               augment=True):
    """
    Preprocess dataset and save the tokenized version with train/validation/test splits.
    Also saves just the training split to a separate directory for accumulating training data.
    
    Args:
        json_path: Path to JSON file with NER data
        save_path: Path where to save the tokenized dataset
        is_testing: If True, all data will be placed in the test split for evaluation
        ds_name: Name to use when saving just the training split (default: "default")
        augment: Whether to apply data augmentation (default: True)
    """
    data = load_dataset(json_path)
    
    # Fix invalid BIO sequences
    data = fix_bio_sequences(data)
    
    # Split into manageable chunks
    data = split_text_into_manageable_chunks(data)
    
    # Add data filtering for better balance
    if not is_testing:
        person_examples = []
        non_person_examples = []
        
        for example in data:
            if any(tag in [LABELS["B-PERSON"], LABELS["I-PERSON"]] for tag in example["ner_tags"]):
                person_examples.append(example)
            else:
                non_person_examples.append(example)
        
        # Keep all person examples, but sample from non-person examples
        # Adjust ratio as needed: 2 means 2 non-person examples for each person example
        ratio = 3  
        non_person_count = min(len(person_examples) * ratio, len(non_person_examples))
        
        import random
        sampled_non_person = random.sample(non_person_examples, non_person_count)
        
        print(f"Data filtering: Keeping {len(person_examples)} examples with person entities")
        print(f"Data filtering: Sampling {non_person_count} from {len(non_person_examples)} examples without person entities")
        
        # Combine and shuffle
        balanced_data = person_examples + sampled_non_person
        random.shuffle(balanced_data)
        data = balanced_data
    
    # Create dataset from the data
    dataset = Dataset.from_list(data)
    data = split_text_into_manageable_chunks(data)
    dataset = Dataset.from_list(data)
    
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
        
        # Apply data augmentation to training split only
        if augment:
            print("Applying data augmentation to training split...")
            augmented_train = augment_dataset(split_dataset["train"])
            split_dataset["train"] = Dataset.from_list(augmented_train)
            print(f"Training split size after augmentation: {len(split_dataset['train'])} examples")
        
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