import os
import torch
import numpy as np
import sys
import random
import json
from tqdm import tqdm
from datasets import load_from_disk
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    PreTrainedModel,
    PretrainedConfig
)
from sklearn.metrics import classification_report
from seqeval.metrics import classification_report as seq_classification_report
from seqeval.scheme import IOB2
import time  # Add this import at the top
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
from torch.nn import CrossEntropyLoss
from torchcrf import CRF
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    DATA_DIR,
    MODEL_NAME,
    LABEL2ID,
    ID2LABEL,
    MODEL_OUTPUT_DIR,
    DATASET_PATH,
    NUM_LABELS,
    LOGS_DIR,
    PROJECT_DIR
)
from utils.label_config import get_label_config, save_label_config, load_label_config, detect_label_schema
from torch.utils.data import DataLoader
import math 
from datasets import concatenate_datasets

class RobertaCRFConfig(PretrainedConfig):
    """Configuration class for RobertaCRF models"""
    model_type = "roberta_crf"
    
    def __init__(
        self,
        vocab_size=50265,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        num_labels=3,
        id2label=None,
        label2id=None,
        crf_weight=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_labels = num_labels
        self.id2label = id2label or {str(i): f"LABEL_{i}" for i in range(num_labels)}
        self.label2id = label2id or {f"LABEL_{i}": i for i in range(num_labels)}
        self.crf_weight = crf_weight

class SimplifiedRobertaCRFForTokenClassification(PreTrainedModel):
    config_class = RobertaCRFConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel.from_pretrained(
            "roberta-base", 
            ignore_mismatched_sizes=True,
            add_pooling_layer=False
        )
        
        # Simple dropout and classifier
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)
        self.crf_weight = getattr(config, 'crf_weight', 1.0)
          # Initialize weights
        self.post_init()
    
    @classmethod
    def from_pretrained_custom(cls, model_name_or_path, num_labels, id2label, label2id, **kwargs):
        """Custom factory method to create model with proper config"""
        config = RobertaCRFConfig(
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            **kwargs
        )
        return cls(config)
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs[0])
        emissions = self.classifier(sequence_output)
        
        if labels is not None:
            # Training mode - simple CRF loss only
            labels_mask = labels >= 0
            crf_mask = attention_mask.bool() & labels_mask
            
            # Ensure first timestep is valid
            for i in range(crf_mask.shape[0]):
                if crf_mask[i].sum() > 0:
                    crf_mask[i, 0] = True
            
            crf_labels = labels.clone()
            crf_labels[~labels_mask] = 0
            
            try:
                loss = -self.crf(emissions, crf_labels, mask=crf_mask, reduction='mean')
            except Exception as e:
                # Fallback to cross-entropy
                active_loss = attention_mask.view(-1) == 1
                active_loss = active_loss & (labels.view(-1) >= 0)
                active_logits = emissions.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = F.cross_entropy(active_logits, active_labels)
            
            return {"loss": loss, "logits": emissions}
        else:
            # Inference mode - Use argmax for consistency with Trainer
            predictions = torch.argmax(emissions, dim=-1)
            return {"logits": emissions, "predictions": predictions}
    
    def _convert_crf_to_tensor(self, crf_predictions, shape):
        """Convert CRF decode output (list of lists) to tensor format"""
        batch_size, seq_length = shape
        predictions_tensor = torch.zeros(batch_size, seq_length, dtype=torch.long, device=self.roberta.device)
        
        for i, pred_seq in enumerate(crf_predictions):
            seq_len = min(len(pred_seq), seq_length)
            predictions_tensor[i, :seq_len] = torch.tensor(pred_seq[:seq_len], dtype=torch.long)
        
        return predictions_tensor
    
    def predict_with_crf(self, input_ids, attention_mask):
        """Separate method for CRF-based inference when needed"""
        self.eval()
        with torch.no_grad():
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = self.dropout(outputs[0])
            emissions = self.classifier(sequence_output)
            
            # Use CRF decode for best predictions
            crf_predictions = self.crf.decode(emissions, mask=attention_mask.bool())
            return crf_predictions

def extract_person_names_from_dataset(dataset):
    """
    Extract person names from the dataset to build a dictionary for augmentation
    """
    first_names = []
    last_names = []
    
    # Process each example in the dataset
    for example in dataset:
        # FIX: Check for different possible field names in your dataset
        tokens = example.get("tokens") or example.get("input_ids") or example.get("text", [])
        ner_tags = example.get("ner_tags") or example.get("labels", [])
        
        # If tokens are IDs, we need to convert them back to text
        # For now, skip examples that don't have proper token structure
        if not isinstance(tokens, list) or not isinstance(ner_tags, list):
            continue
            
        # If tokens are actually token IDs, we'd need a tokenizer to decode them
        # For this fix, we'll skip ID-based tokens and only process text tokens
        if tokens and isinstance(tokens[0], int):
            continue  # Skip tokenized data for now
        
        i = 0
        while i < len(tokens) and i < len(ner_tags):
            # Look for B-PERSON tags
            if i < len(ner_tags) and str(ner_tags[i]) in ID2LABEL and ID2LABEL[str(ner_tags[i])] == "B-PERSON":
                name_tokens = []
                
                # Safely collect the name tokens
                while (i < len(tokens) and i < len(ner_tags) and 
                       str(ner_tags[i]) in ID2LABEL and 
                       (ID2LABEL[str(ner_tags[i])] == "B-PERSON" or 
                        ID2LABEL[str(ner_tags[i])] == "I-PERSON")):
                    name_tokens.append(tokens[i])
                    i += 1
                
                # Process the collected name
                if name_tokens:
                    # Single-token names are likely first names
                    if len(name_tokens) == 1:
                        first_names.append(name_tokens[0])
                    # Multi-token names - first token is first name, last token is last name
                    elif len(name_tokens) > 1:
                        first_names.append(name_tokens[0])
                        last_names.append(name_tokens[-1])
            else:
                i += 1
    
    # Return dictionary of names with fallbacks
    return {
        "first_names": list(set(first_names)) or ["John", "Jane", "Michael", "Sarah", "David"],
        "last_names": list(set(last_names)) or ["Smith", "Johnson", "Brown", "Davis", "Wilson"]
    }
def augment_person_entities(dataset, augmentation_factor=0.3):
    """
    Augment the dataset with additional person entities
    """
    print("Starting data augmentation...")
    
    # Try to load names from the dictionary file
    json_path = os.path.join(DATA_DIR, 'names_dictionary.json')
    names_dict = None
    
    try:
        # Try to load existing dictionary
        with open(json_path, 'r', encoding='utf-8') as f:
            names_dict = json.load(f)
        print(f"Loaded names dictionary with {sum(len(v) for v in names_dict.values())} names")
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Names dictionary not found or invalid at {json_path}")
        # Create names dictionary from the dataset
        try:
            names_dict = extract_person_names_from_dataset(dataset)
            
            # Save for future use if we have names
            if names_dict and any(len(v) > 0 for v in names_dict.values()):
                os.makedirs(os.path.dirname(json_path), exist_ok=True)
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(names_dict, f, indent=2)
                print(f"Created new names dictionary with {sum(len(v) for v in names_dict.values())} names")
            else:
                print("Could not extract enough names for augmentation")
                return dataset  # Return original dataset if we can't extract names
        except Exception as e:
            print(f"Error extracting names from dataset: {e}")
            return dataset  # Return original dataset on error
    
    # Extract first and last names from the dictionary
    first_names = names_dict.get("first_names", [])
    last_names = names_dict.get("last_names", [])
    
    # Skip augmentation if we don't have names
    if not first_names or not last_names:
        print("Not enough names for augmentation, using fallback names")
        first_names = first_names or ["John", "Jane", "Michael", "Sarah", "David"]
        last_names = last_names or ["Smith", "Johnson", "Brown", "Davis", "Wilson"]
        
    # Identify sequences with person entities
    def find_person_spans(labels):
        spans = []
        current_span = None
        
        for i, label in enumerate(labels):
            if label == LABEL2ID["B-PERSON"]:
                if current_span is not None:
                    spans.append(current_span)
                current_span = {"start": i, "end": i, "type": "PERSON"}
            elif label == LABEL2ID["I-PERSON"] and current_span is not None:
                current_span["end"] = i
            elif current_span is not None:
                spans.append(current_span)
                current_span = None
                
        if current_span is not None:
            spans.append(current_span)
        
        return spans
    
    # Load the tokenizer to tokenize new names
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print("Augmenting dataset with person name variations...")
    augmented_examples = []
    num_to_augment = int(len(dataset) * augmentation_factor)
    
    # Select examples with person entities
    examples_with_person = []
    for example in dataset:
        spans = find_person_spans(example["labels"])
        if spans:
            examples_with_person.append((example, spans))
    
    if not examples_with_person:
        print("No examples with person entities found for augmentation")
        return dataset
    
    # Create augmented examples
    for _ in tqdm(range(num_to_augment)):
        # Select a random example with person entities
        example, spans = random.choice(examples_with_person)
        
        # Create a copy of the example
        new_example = {k: v.copy() if isinstance(v, list) else v for k, v in example.items()}
        
        # For each person entity, replace with a random name
        for span in spans:
            # Get span length
            span_length = span["end"] - span["start"] + 1
            
            # Decide if we'll use first name only or first+last
            if random.random() < 0.5 or span_length <= 1:
                new_name = random.choice(first_names)
            else:
                new_name = f"{random.choice(first_names)} {random.choice(last_names)}"
            
            # Tokenize the new name
            name_tokens = tokenizer.tokenize(new_name)
            name_ids = tokenizer.convert_tokens_to_ids(name_tokens)
            
            # Get current span length in the example
            original_span_length = span["end"] - span["start"] + 1
            new_span_length = len(name_ids)
            
            # If the new name has the same number of tokens, simple replacement
            if new_span_length == original_span_length:
                # Replace input IDs
                new_example["input_ids"][span["start"]:span["end"]+1] = name_ids
                
                # Update labels (first token is B-PERSON, rest are I-PERSON)
                new_example["labels"][span["start"]] = LABEL2ID["B-PERSON"]
                for i in range(span["start"]+1, span["end"]+1):
                    new_example["labels"][i] = LABEL2ID["I-PERSON"]
            
            # If new name has fewer tokens, we need to pad
            elif new_span_length < original_span_length:
                # Replace input IDs
                for i in range(new_span_length):
                    new_example["input_ids"][span["start"] + i] = name_ids[i]
                
                # Pad the rest with the last token ID of input_ids
                pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                for i in range(new_span_length, original_span_length):
                    new_example["input_ids"][span["start"] + i] = pad_token
                    new_example["labels"][span["start"] + i] = LABEL2ID["O"]  # Change to O
                    new_example["attention_mask"][span["start"] + i] = 0  # Mask out padding
                
                # Update labels for the actual name tokens
                new_example["labels"][span["start"]] = LABEL2ID["B-PERSON"]
                for i in range(1, new_span_length):
                    new_example["labels"][span["start"] + i] = LABEL2ID["I-PERSON"]
            
            # If new name has more tokens, we need to truncate
            else:  # new_span_length > original_span_length
                # Replace input IDs (use as many as fit)
                for i in range(original_span_length):
                    new_example["input_ids"][span["start"] + i] = name_ids[i]
                
                # Update labels
                new_example["labels"][span["start"]] = LABEL2ID["B-PERSON"]
                for i in range(1, original_span_length):
                    new_example["labels"][span["start"] + i] = LABEL2ID["I-PERSON"]
        
        # Add to augmented examples
        augmented_examples.append(new_example)
    
    print(f"Added {len(augmented_examples)} augmented examples")
    
    # FIX: Properly concatenate HuggingFace datasets
    if augmented_examples:
        # Import the necessary modules for dataset creation
        from datasets import Dataset
        
        # Create a new dataset from augmented examples
        augmented_dataset = Dataset.from_list(augmented_examples)
        
        # Use the datasets library concatenate_datasets function

        combined_dataset = concatenate_datasets([dataset, augmented_dataset])
        
        return combined_dataset
    else:
        return dataset

def compute_metrics(eval_preds):
    """Compute metrics for evaluation"""
    print(f"Type of eval_preds: {type(eval_preds)}")
    
    try:
        # Handle HuggingFace EvalPrediction object
        if hasattr(eval_preds, 'predictions') and hasattr(eval_preds, 'label_ids'):
            print("HuggingFace EvalPrediction object format")
            preds = eval_preds.predictions
            labels = eval_preds.label_ids
            
            # Handle different prediction formats
            if isinstance(preds, tuple):
                print(f"Predictions is a tuple of length {len(preds)}")
                preds = preds[0]  # Use first element
            
            # Convert to numpy if needed
            if isinstance(preds, torch.Tensor):
                predictions = preds.cpu().numpy()
            else:
                predictions = np.array(preds)
            
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            else:
                labels = np.array(labels)
            
            # Apply argmax if logits
            if len(predictions.shape) == 3:  # [batch, seq, num_classes]
                predictions = np.argmax(predictions, axis=-1)
        else:
            # Handle tuple format
            if isinstance(eval_preds, tuple) and len(eval_preds) == 2:
                preds, labels = eval_preds
                
                if isinstance(preds, tuple):
                    preds = preds[0]
                
                # Convert to numpy
                if isinstance(preds, torch.Tensor):
                    predictions = preds.cpu().numpy()
                else:
                    predictions = np.array(preds)
                
                if isinstance(labels, torch.Tensor):
                    labels = labels.cpu().numpy()
                else:
                    labels = np.array(labels)
                
                # Apply argmax if needed
                if len(predictions.shape) == 3:
                    predictions = np.argmax(predictions, axis=-1)
            else:
                raise ValueError(f"Unexpected eval_preds format: {type(eval_preds)}")
        
        # Extract valid predictions and labels
        true_predictions = []
        true_labels = []
        
        for i in range(len(predictions)):
            pred_seq = []
            label_seq = []
            
            pred_len = len(predictions[i]) if hasattr(predictions[i], '__len__') else 1
            label_len = len(labels[i]) if hasattr(labels[i], '__len__') else 1
            
            for j in range(min(pred_len, label_len)):
                if j < len(labels[i]) and labels[i][j] != -100:
                    pred_id = int(predictions[i][j])
                    label_id = int(labels[i][j])
                    
                    if str(pred_id) in ID2LABEL and str(label_id) in ID2LABEL:
                        pred_seq.append(ID2LABEL[str(pred_id)])
                        label_seq.append(ID2LABEL[str(label_id)])
            
            if pred_seq and label_seq:
                true_predictions.append(pred_seq)
                true_labels.append(label_seq)
        
        if not true_predictions or not true_labels:
            print("Warning: No valid predictions found")
            return {
                "person_f1": 0.0,
                "entity_f1": 0.0,
                "token_accuracy": 0.0
            }
        
        # Calculate metrics
        entity_results = seq_classification_report(
            true_labels, true_predictions, scheme=IOB2, output_dict=True
        )
        
        person_f1 = entity_results.get("PERSON", {}).get("f1-score", 0.0)
        
        # Token-level metrics
        all_true_labels = [l for seq in true_labels for l in seq]
        all_predictions = [p for seq in true_predictions for p in seq]
        
        if all_true_labels and all_predictions:
            token_results = classification_report(
                all_true_labels, all_predictions, output_dict=True
            )
            token_accuracy = token_results["accuracy"]
        else:
            token_accuracy = 0.0
        
        results = {
            "person_f1": person_f1,
            "entity_f1": entity_results["macro avg"]["f1-score"],
            "token_accuracy": token_accuracy
        }
        
        return results
        
    except Exception as e:
        print(f"Critical error in compute_metrics: {e}")
        import traceback
        traceback.print_exc()
        return {
            "person_f1": 0.0,
            "entity_f1": 0.0,
            "token_accuracy": 0.0
        }

 
class CrossAttentionSpanClassifier(nn.Module):
    """Enhanced classifier that uses cross-attention to capture entity spans"""
    def __init__(self, hidden_size, num_labels, num_attention_heads=4, max_relative_position=5):
        super().__init__()
        self.original_hidden_size = hidden_size
        self.num_labels = num_labels
        self.num_attention_heads = num_attention_heads
        
        # Ensure we have a hidden size that's divisible by num_attention_heads
        # Find the largest size <= hidden_size that's divisible by num_attention_heads
        self.attention_hidden_size = (hidden_size // num_attention_heads) * num_attention_heads
        self.attention_head_size = self.attention_hidden_size // num_attention_heads
        self.max_relative_position = max_relative_position
        
        # Add projection layer if sizes don't match
        self.input_projection = None
        if self.attention_hidden_size != hidden_size:
            self.input_projection = nn.Linear(hidden_size, self.attention_hidden_size)
            print(f"Added input projection: {hidden_size} -> {self.attention_hidden_size}")
        
        # Use attention_hidden_size for internal computations
        self.hidden_size = self.attention_hidden_size
          # Standard token classification layer (use original hidden size)
        self.token_classifier = nn.Linear(self.original_hidden_size, num_labels)
        # Cross-attention components (use attention_hidden_size for internal computation)
        self.query = nn.Linear(self.attention_hidden_size, self.attention_hidden_size)
        self.key = nn.Linear(self.attention_hidden_size, self.attention_hidden_size)
        self.value = nn.Linear(self.attention_hidden_size, self.attention_hidden_size)
        
        # Output projection (from attention space back to original space)
        self.output_projection = nn.Linear(self.attention_hidden_size, self.original_hidden_size)
        
        # Layer norm and dropout (use original hidden size)
        self.layer_norm = nn.LayerNorm(self.original_hidden_size)
        self.dropout = nn.Dropout(0.1)
        
        # Relative position embeddings for entity spans
        self.relative_position_embeddings = nn.Parameter(
            torch.randn(2 * max_relative_position + 1, self.attention_head_size)
        )
        
        # Final span-aware classification layer (use original hidden size)
        self.span_classifier = nn.Linear(self.original_hidden_size, num_labels)
        # Entity bias for B-PERSON and I-PERSON to encourage span coherence
        self.entity_bias = nn.Parameter(torch.zeros(num_labels))
        
    def transpose_for_scores(self, x):
        """Reshape for multi-head attention"""
        batch_size, seq_length, _ = x.shape
        x = x.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)  # [batch, heads, seq_len, head_size]
    
    def forward(self, sequence_output):
        batch_size, seq_length, _ = sequence_output.shape
        
        # Apply input projection if needed to match attention dimensions
        if self.input_projection is not None:
            projected_output = self.input_projection(sequence_output)
        else:
            projected_output = sequence_output
        
        # Base token classification logits (using original sequence_output)
        base_logits = self.token_classifier(sequence_output)
        
        # Compute query, key, value projections using projected_output
        query_layer = self.transpose_for_scores(self.query(projected_output))
        key_layer = self.transpose_for_scores(self.key(projected_output))
        value_layer = self.transpose_for_scores(self.value(projected_output))
        
        # Compute attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
          # Add relative position information to favor token relationships within entity spans
        # This helps the model recognize that tokens close to each other are more likely
        # to belong to the same entity
        rel_pos_scores = self.compute_relative_positions(seq_length)
        attention_scores = attention_scores + rel_pos_scores  # Broadcasting: [batch, heads, seq, seq] + [seq, seq]
        
        # Apply distance-based attention mask that favors nearby tokens
        # (making it easier to identify entity spans)
        distance_mask = self.compute_distance_mask(seq_length)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + distance_mask  # Broadcasting: [batch, heads, seq, seq] + [seq, seq]
        
        # Softmax attention
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
          # Apply attention to values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(batch_size, seq_length, self.attention_hidden_size)
        
        # Post-process with layer norm and residual connection
        # output_projection maps from attention space back to original space
        projected_context = self.output_projection(context_layer)
        output_hidden_states = self.layer_norm(sequence_output + projected_context)
        
        # Get span-aware classification logits
        span_logits = self.span_classifier(output_hidden_states)
          # Apply entity bias to encourage span coherence
        # (increases probability of I-PERSON following B-PERSON)
        final_logits = span_logits.clone()
        
        # Apply entity coherence bias
        for i in range(batch_size):
            for j in range(1, seq_length):
                # If previous token was predicted as B-PERSON, boost I-PERSON probability
                if torch.argmax(span_logits[i, j-1]) == LABEL2ID["B-PERSON"]:
                    final_logits[i, j, LABEL2ID["I-PERSON"]] += self.entity_bias[LABEL2ID["I-PERSON"]]
                # If previous token was I-PERSON, also boost I-PERSON                elif torch.argmax(span_logits[i, j-1]) == LABEL2ID["I-PERSON"]:
                    final_logits[i, j, LABEL2ID["I-PERSON"]] += self.entity_bias[LABEL2ID["I-PERSON"]]        
        return final_logits
    
    def compute_relative_positions(self, seq_length):
        """Compute relative position scores for attention with fixed dimensions"""
        # Simplify this to avoid tensor shape issues
        device = self.relative_position_embeddings.device
        
        # Create a simple position bias that decays with distance
        # This avoids complex embedding lookups that can cause shape mismatches
        range_vec = torch.arange(seq_length, device=device, dtype=torch.float)
        range_mat = range_vec.unsqueeze(0) - range_vec.unsqueeze(1)  # [seq_len, seq_len]
        
        # Apply decay based on distance, clamped to max_relative_position
        distance = torch.abs(range_mat)
        clamped_distance = torch.clamp(distance, 0, self.max_relative_position)
        
        # Simple exponential decay: closer tokens get higher scores
        position_bias = torch.exp(-0.1 * clamped_distance)  # [seq_len, seq_len]
          # Return without expanding - let broadcasting handle it
        return position_bias
    
    def compute_distance_mask(self, seq_length):
        """Create attention mask that favors nearby tokens"""
        device = self.relative_position_embeddings.device
        
        # Create position indices matrix
        range_vec = torch.arange(seq_length, device=device, dtype=torch.float)
        distance_mat = torch.abs(range_vec.unsqueeze(0) - range_vec.unsqueeze(1))
        
        # Convert distances to mask values (closer tokens get less penalty)
        attn_mask = -0.1 * distance_mat  # [seq_len, seq_len]
        
        # Return without expanding - let broadcasting handle it
        return attn_mask
    
class RobertaCRFForTokenClassification(nn.Module):
    def __init__(self, model_name, num_labels, alpha=0.25, gamma=2, person_weight=2.0, 
                 crf_weight=0.6, focal_weight=0.2, dice_weight=0.2,
                 classifier_params=None, dice_loss_params=None):
        super().__init__()
        self.num_labels = num_labels
        
        # Add ignore_mismatched_sizes=True to handle size mismatches
        self.roberta = RobertaModel.from_pretrained(
            model_name,
            ignore_mismatched_sizes=True  # Add this parameter
        )
        
        # Get config from the base model
        self.config = self.roberta.config
          # Add token classification specific attributes to config
        self.config.num_labels = num_labels
        self.config.id2label = ID2LABEL
        self.config.label2id = LABEL2ID
        self.config.problem_type = "token_classification"
        
        # Add custom model parameters to config for proper saving/loading
        self.config.alpha = alpha
        self.config.gamma = gamma
        self.config.person_weight = person_weight
        self.config.crf_weight = crf_weight
        self.config.focal_weight = focal_weight
        self.config.dice_weight = dice_weight
        self.config.classifier_params = classifier_params or {}
        self.config.dice_loss_params = dice_loss_params or {}
        
        # Default parameters
        classifier_params = classifier_params or {}
        dice_loss_params = dice_loss_params or {}
        
        # Model components
        self.dropout = nn.Dropout(classifier_params.get("dropout", 0.1))
         
        # Create the cross-attention span classifier with tunable parameters
        self.classifier = CrossAttentionSpanClassifier(
            self.roberta.config.hidden_size, 
            num_labels,
            num_attention_heads=classifier_params.get("num_attention_heads", 4),
            max_relative_position=classifier_params.get("max_relative_position", 5)
        )
        
        self.crf = CRF(num_labels, batch_first=True)
        
        # Loss parameters
        self.alpha = alpha
        self.gamma = gamma
        self.person_weight = person_weight
        self.crf_weight = crf_weight
        self.focal_weight = focal_weight
        
        # Enhanced boundary-focused loss with tunable parameters
        self.dice_loss = EnhancedBoundaryDiceLoss(
            smooth=dice_loss_params.get("smooth", 1e-5),
            b_weight=dice_loss_params.get("b_weight", 3.0),
            i_end_weight=dice_loss_params.get("i_end_weight", 2.5),
            context_weight=dice_loss_params.get("context_weight", 1.5)
        )
        self.dice_weight = dice_weight
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass for token classification with CRF, focal loss, and dice loss
        """
        # Filter out unexpected kwargs that shouldn't be passed to the RobertaModel
        roberta_kwargs = {k: v for k, v in kwargs.items() if k not in [
            'num_items_in_batch',  # Filter out this problematic parameter
            'return_dict',         # Handle these specially if needed
            'output_hidden_states',
            'output_attentions'
        ]}
        
        # Get the base model outputs
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **roberta_kwargs
        )
        
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        
        # Get logits from classifier (could be standard or span-aware)
        emissions = self.classifier(sequence_output)
        
        if labels is not None:
            # Training mode
            # Create a mask for valid positions (not -100)
            labels_mask = labels >= 0
            
            # Apply mask to both attention_mask and labels
            crf_mask = attention_mask.bool() & labels_mask
            
            # FIX: Ensure first timestep is always masked as valid for CRF
            # CRF requires the first token to be valid
            for i in range(crf_mask.shape[0]):
                if crf_mask[i].sum() > 0:  # If there are any valid tokens
                    # Find first valid position and ensure it's marked as valid
                    first_valid = crf_mask[i].nonzero(as_tuple=True)[0]
                    if len(first_valid) > 0:
                        first_idx = first_valid[0].item()
                        if first_idx > 0:
                            # Shift mask to ensure first position is valid
                            crf_mask[i, 0] = True
                            if first_idx < crf_mask.shape[1]:
                                crf_mask[i, first_idx] = False
            
            # Replace -100 with 0 for CRF calculation
            crf_labels = labels.clone()
            crf_labels[~labels_mask] = 0
            
            # Calculate CRF loss (sequence level) with proper mask
            try:
                crf_loss = -self.crf(emissions, crf_labels, mask=crf_mask, reduction='mean')
            except Exception as e:
                print(f"CRF loss calculation failed: {e}, using fallback")
                # Fallback to cross-entropy loss
                active_loss = attention_mask.view(-1) == 1
                active_loss = active_loss & (labels.view(-1) >= 0)
                active_logits = emissions.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                crf_loss = F.cross_entropy(active_logits, active_labels)
            
            # Calculate token-level focal loss on valid positions
            active_loss = attention_mask.view(-1) == 1
            active_loss = active_loss & (labels.view(-1) >= 0)
            active_logits = emissions.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            
            if len(active_labels) > 0:
                # Calculate focal loss with parameterized person weight
                fl_loss = focal_loss(active_logits, active_labels, 
                                    alpha=self.alpha, gamma=self.gamma, 
                                    person_weight=self.person_weight)
                
                # Calculate Dice loss
                dice_loss = self.dice_loss(active_logits, active_labels)
                
                # Combined weighted loss
                loss = (self.crf_weight * crf_loss + 
                        self.focal_weight * fl_loss + 
                        self.dice_weight * dice_loss)
            else:
                loss = crf_loss
                
            return {"loss": loss, "logits": emissions}
        else:
            # Inference mode - use argmax instead of CRF for simplicity in evaluation
            predictions = torch.argmax(emissions, dim=2)
            
            # Apply confidence-based post-processing with safe dimension handling
            if attention_mask is not None:
                # Get minimum valid length for all tensors
                mask_length = min(attention_mask.size(1), predictions.size(1), emissions.size(1))
                
                # Only process what fits
                if mask_length > 0:
                    processed_predictions = confidence_based_postprocessing(
                        emissions[:, :mask_length, :], 
                        predictions[:, :mask_length],
                        attention_mask[:, :mask_length]
                    )
                    return {
                        "logits": emissions,
                        "predictions": processed_predictions,
                        "raw_predictions": predictions
                    }
            
            return {"logits": emissions, "predictions": predictions}
def focal_loss(logits, labels, alpha=0.25, gamma=2, person_weight=2.0):
    """
    Focal loss for multi-class classification with extra weight on minority classes.
    - alpha: balancing parameter (higher for minority classes)
    - gamma: focusing parameter (higher means more focus on hard examples)
    - person_weight: specific weight for PERSON tags
    """
    ce_loss = F.cross_entropy(logits, labels, reduction='none', ignore_index=-100)
    pt = torch.exp(-ce_loss)
    
    # Apply class weights here with parameterized person weight
    alpha_weight = torch.ones_like(labels).float()
    
    # Safely apply weights only to classes that exist in our label set
    person_mask = (labels == LABEL2ID.get("B-PERSON", -1)) | (labels == LABEL2ID.get("I-PERSON", -1))
    alpha_weight[person_mask] = person_weight  # Parameterized person weight
    
    # Only apply title weights if those labels exist in our schema
    if "B-TITLE" in LABEL2ID and "I-TITLE" in LABEL2ID:
        title_mask = (labels == LABEL2ID["B-TITLE"]) | (labels == LABEL2ID["I-TITLE"])
        alpha_weight[title_mask] = 1.5   
    
    focal_loss = alpha_weight * ((1 - pt) ** gamma) * ce_loss
    return focal_loss.mean()


confidence_threshold = 0.7

def confidence_based_postprocessing(logits, predictions, attention_mask):
    """Apply confidence thresholds to improve boundary decisions"""
    batch_size, seq_length, num_classes = logits.shape
    
    # Use the global confidence threshold
    global confidence_threshold
    
    # Get softmax probabilities
    probs = F.softmax(logits, dim=2)
    processed_preds = predictions.clone()
    
    for i in range(batch_size):
        for j in range(1, seq_length-1):  # Skip first and last tokens
            if attention_mask[i, j] == 0:  # Skip padding
                continue
                
            # Get current and surrounding token predictions
            current_pred = predictions[i, j].item()
            prev_pred = predictions[i, j-1].item() if j > 0 else -1
            next_pred = predictions[i, j+1].item() if j < seq_length-1 else -1
            
            # Get confidence scores
            current_conf = probs[i, j, current_pred].item()
            
            # Case 1: Low confidence O between entity tags might be I-PERSON
            if (current_pred == LABEL2ID["O"] and
                current_conf < confidence_threshold and
                (prev_pred == LABEL2ID["B-PERSON"] or prev_pred == LABEL2ID["I-PERSON"]) and
                (next_pred == LABEL2ID["I-PERSON"] or next_pred == LABEL2ID["B-PERSON"])):
                
                # Change to I-PERSON
                processed_preds[i, j] = LABEL2ID["I-PERSON"]
            
            # Case 2: Low confidence I-PERSON after O should be B-PERSON
            elif (current_pred == LABEL2ID["I-PERSON"] and
                  current_conf < confidence_threshold * 1.1 and  # Slightly higher threshold
                  prev_pred == LABEL2ID["O"]):
                
                # Change to B-PERSON
                processed_preds[i, j] = LABEL2ID["B-PERSON"]
            
            # Case 3: Very low confidence B-PERSON between two O tags
            elif (current_pred == LABEL2ID["B-PERSON"] and
                  current_conf < confidence_threshold * 0.8 and  # Lower threshold
                  prev_pred == LABEL2ID["O"] and 
                  next_pred == LABEL2ID["O"]):
                
                # This might be a false positive - change to O
                processed_preds[i, j] = LABEL2ID["O"]
                
            # Case 4: Isolated I-PERSON (not following B-PERSON or I-PERSON)
            elif (current_pred == LABEL2ID["I-PERSON"] and
                  prev_pred != LABEL2ID["B-PERSON"] and
                  prev_pred != LABEL2ID["I-PERSON"]):
                
                # Fix invalid sequence - change to B-PERSON
                processed_preds[i, j] = LABEL2ID["B-PERSON"]
    
    return processed_preds

class EnhancedBoundaryDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, b_weight=3.0, i_end_weight=2.5, context_weight=1.5):
        super().__init__()
        self.smooth = smooth
        self.b_weight = b_weight  
        self.i_end_weight = i_end_weight  
        self.context_weight = context_weight  
        
    def forward(self, inputs, targets):
        # Get probabilities
        probs = F.softmax(inputs, dim=1)
        
        # One-hot encode targets
        num_classes = inputs.size(1)
        targets_one_hot = F.one_hot(targets, num_classes).float()
        
        # Initialize weights tensor - base weight 1.0
        weights = torch.ones_like(targets).float()
        
        # 1. B-PERSON gets highest weight
        weights[targets == LABEL2ID["B-PERSON"]] = self.b_weight
        
        # 2. Identify entity end boundaries (last I-PERSON in sequence)
        if len(targets.shape) == 1:
            # Flattened batch case
            shifted = torch.cat([targets[1:], targets[-1:]])
            is_i_person = (targets == LABEL2ID["I-PERSON"])
            next_not_i_person = (shifted != LABEL2ID["I-PERSON"])
            # Final I-PERSON in a sequence
            weights[is_i_person & next_not_i_person] = self.i_end_weight
        else:
            # Batch dimension case
            for i in range(targets.shape[0]):
                for j in range(len(targets[i])-1):
                    # Final I-PERSON (followed by non-I-PERSON)
                    if (targets[i,j] == LABEL2ID["I-PERSON"] and 
                        targets[i,j+1] != LABEL2ID["I-PERSON"]):
                        weights[i,j] = self.i_end_weight
                
                # Handle end of sequence
                if targets[i,-1] == LABEL2ID["I-PERSON"]:
                    weights[i,-1] = self.i_end_weight
        
        # 3. Boost tokens immediately before B-PERSON (context awareness)
        if len(targets.shape) == 1:
            # Find tokens before B-PERSON
            shifted_back = torch.cat([torch.tensor([0], device=targets.device), targets[:-1]])
            before_b = (targets == LABEL2ID["B-PERSON"]) & (shifted_back != -100)
            # Apply weight to token before B-PERSON
            if before_b.any():
                indices = torch.nonzero(before_b, as_tuple=True)[0]
                indices = indices - 1  # Get previous token
                valid_indices = indices[indices >= 0]  # Ensure non-negative indices
                weights[valid_indices] = self.context_weight
        else:
            # Handle batch dimension
            for i in range(targets.shape[0]):
                for j in range(1, len(targets[i])):
                    if targets[i,j] == LABEL2ID["B-PERSON"]:
                        # Weight token before B-PERSON
                        weights[i,j-1] = self.context_weight
        
        # Handle dimension expansion for weights
        weights_expanded = weights.unsqueeze(-1) if len(weights.shape) < len(targets_one_hot.shape) else weights
        
        # Weighted Dice calculation
        intersection = (probs * targets_one_hot * weights_expanded).sum(dim=0)
        denominator = ((probs * weights_expanded).sum(dim=0) + 
                      (targets_one_hot * weights_expanded).sum(dim=0))
            
        # Dice coefficient per class
        dice_coef = (2 * intersection + self.smooth) / (denominator + self.smooth)
        
        # Calculate loss
        dice_loss = 1 - dice_coef.mean()
        
        return dice_loss

class ContextAwareSpanClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels, max_span_length=5, context_ratio=0.5):
        super().__init__()
        self.max_span_length = max_span_length
        self.hidden_size = hidden_size
        self.context_ratio = context_ratio
        
        # Main classification layer
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # Span attention components
        self.span_query = nn.Parameter(torch.randn(hidden_size))
        self.span_attention = nn.Linear(hidden_size, 1)
        
        # Context enhancement
        self.context_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_labels)
        )
        
    def forward(self, sequence_output):
        batch_size, seq_length, hidden_size = sequence_output.shape
        
        # Standard token classification logits
        base_logits = self.classifier(sequence_output)
        
        # Enhanced logits with context awareness
        enhanced_logits = torch.zeros_like(base_logits)
        
        # Process each example in batch
        for b in range(batch_size):
            # For each position, consider a surrounding window
            for i in range(seq_length):
                # Define context window
                left_ctx = max(0, i - self.max_span_length)
                right_ctx = min(seq_length, i + self.max_span_length + 1)
                
                if right_ctx - left_ctx <= 1:  # Skip if window too small
                    enhanced_logits[b, i] = base_logits[b, i]
                    continue
                
                # Extract context window
                ctx_tokens = sequence_output[b, left_ctx:right_ctx]
                
                # Calculate attention scores within window
                attn_scores = self.span_attention(ctx_tokens).squeeze(-1)
                attn_weights = F.softmax(attn_scores, dim=0)
                
                # Calculate weighted context
                weighted_ctx = (ctx_tokens * attn_weights.unsqueeze(-1)).sum(dim=0)
                
                # Combine current token with context
                token_with_ctx = torch.cat([
                    sequence_output[b, i],  # Current token
                    weighted_ctx            # Context representation
                ], dim=-1)
                
                # Get context-aware logits
                ctx_logits = self.context_layer(token_with_ctx)
                
                # Blend base and context logits
                enhanced_logits[b, i] = (1 - self.context_ratio) * base_logits[b, i] + \
                                        self.context_ratio * ctx_logits
        
        return enhanced_logits

def evaluate_postprocessing_impact(model, dataset, tokenizer):
    """
    Evaluate the impact of confidence-based post-processing
    by comparing raw model predictions vs processed predictions
    """
    print("\n=== Evaluating Post-Processing Impact ===")
    
    # Determine device from model's parameters instead of assuming model.device exists
    device = next(model.parameters()).device
    
    # Set up data loader
    eval_dataset = dataset
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # Use batch size 16 for evaluation
    dataloader = DataLoader(
        eval_dataset, 
        batch_size=16, 
        collate_fn=data_collator
    )
    
    # Collect predictions with and without post-processing
    all_labels = []
    raw_predictions = []
    processed_predictions = []
    
    # Disable gradient computation for evaluation
    model.eval()  # Ensure model is in eval mode
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating post-processing"):
            # Move batch to appropriate device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # FIX: Ensure proper mask format for CRF
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            
            # Skip batches where any sequence has all zeros in attention mask
            # or where the first token is masked
            valid_batch = True
            for i in range(attention_mask.shape[0]):
                if attention_mask[i, 0] == 0 or attention_mask[i].sum() == 0:
                    valid_batch = False
                    break
            
            if not valid_batch:
                print("Skipping batch with invalid attention mask")
                continue
            
            try:
                # Get model predictions with fixed attention mask
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=attention_mask,
                    labels=None  # Don't pass labels to avoid training mode
                )
                
                logits = outputs["logits"]
                
                # Get raw predictions (without post-processing)
                # Use argmax instead of CRF for raw predictions to avoid mask issues
                raw_preds = torch.argmax(logits, dim=2)
                
                # Apply post-processing to get improved predictions
                processed_preds = confidence_based_postprocessing(
                    logits, raw_preds, attention_mask
                )
                
                # Extract valid labels (not -100)
                valid_mask = labels >= 0
                
                # Only keep predictions for valid positions
                for i in range(labels.size(0)):  # Iterate through batch
                    valid_indices = valid_mask[i].nonzero(as_tuple=True)[0]
                    
                    if len(valid_indices) > 0:  # Ensure there are valid indices
                        # Extract valid portions
                        batch_labels = labels[i, valid_indices].cpu().numpy()
                        batch_raw_preds = raw_preds[i, valid_indices].cpu().numpy()
                        batch_processed_preds = processed_preds[i, valid_indices].cpu().numpy()
                        
                        # Convert to label strings
                        label_strs = [ID2LABEL.get(str(label_id), "O") for label_id in batch_labels]
                        raw_pred_strs = [ID2LABEL.get(str(pred_id), "O") for pred_id in batch_raw_preds]
                        processed_pred_strs = [ID2LABEL.get(str(pred_id), "O") for pred_id in batch_processed_preds]
                        
                        all_labels.append(label_strs)
                        raw_predictions.append(raw_pred_strs)
                        processed_predictions.append(processed_pred_strs)
                        
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                continue
    
    if not all_labels:
        print("No valid predictions collected for post-processing evaluation")
        return {
            "raw_f1": 0.0,
            "processed_f1": 0.0,
            "improvement": 0.0
        }
    
    try:
        # Calculate entity-level metrics for raw predictions
        raw_results = seq_classification_report(
            all_labels, raw_predictions, scheme=IOB2, output_dict=True
        )
        
        # Calculate entity-level metrics for processed predictions
        processed_results = seq_classification_report(
            all_labels, processed_predictions, scheme=IOB2, output_dict=True
        )
        
        # Extract F1 scores for PERSON entity
        raw_f1 = raw_results.get("PERSON", {}).get("f1-score", 0.0)
        processed_f1 = processed_results.get("PERSON", {}).get("f1-score", 0.0)
        
        print(f"Raw F1 score for PERSON: {raw_f1:.4f}")
        print(f"Processed F1 score for PERSON: {processed_f1:.4f}")
        print(f"Improvement: {processed_f1 - raw_f1:.4f}")
        
        return {
            "raw_f1": raw_f1,
            "processed_f1": processed_f1,
            "improvement": processed_f1 - raw_f1
        }
        
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return {
            "raw_f1": 0.0,
            "processed_f1": 0.0,
            "improvement": 0.0
        }

def main(config=None, test_mode=False, output_dir=None, dataset_path=DATASET_PATH, model_name=MODEL_NAME, use_simplified_model=False, include_title_tags=None):
    """
    Train NER model with hyperparameter configuration.
    
    Args:
        config: Dictionary containing hyperparameters
        test_mode: Whether to run in test mode with small data
        output_dir: Directory to save model outputs (overrides default)
        use_simplified_model: Whether to use the simplified RoBERTa+CRF model instead of the complex one
        include_title_tags: Whether to include TITLE tags (None=auto-detect from dataset)
    """
    # Detect label schema from dataset if not specified
    if include_title_tags is None:
        print("Auto-detecting label schema from dataset...")
        include_title_tags = detect_label_schema(dataset_path)
        print(f"Detected schema: {'WITH TITLE' if include_title_tags else 'NO TITLE'} tags")
    
    # Get appropriate label configuration
    label_config = get_label_config(include_title_tags)
    
    # Update global label variables for this training session
    global LABEL2ID, ID2LABEL, NUM_LABELS
    LABEL2ID = label_config["label2id"]
    ID2LABEL = label_config["id2label"]
    NUM_LABELS = label_config["num_labels"]
    
    print(f"Using label configuration:")
    print(f"  Labels: {label_config['labels']}")
    print(f"  Number of labels: {NUM_LABELS}")
    
    # Default hyperparameters
    default_config = {
        # Model architecture
        "num_attention_heads": 4,
        "max_relative_position": 5,
        "dropout": 0.1,
        
        # Loss weights
        "crf_weight": 0.5,
        "focal_weight": 0.2,
        "dice_weight": 0.3,
        
        # Focal loss parameters
        "alpha": 0.25,
        "gamma": 2.0,
        "person_weight": 5.0,
        
        # Dice loss parameters
        "dice_smooth": 1e-5,
        "b_weight": 3.0,
        "i_end_weight": 2.5,
        "context_weight": 1.5,
        
        # Post-processing
        "confidence_threshold": 0.7,
        
        # Training parameters
        "learning_rate": 3e-5,
        "weight_decay": 0.01,
        "batch_size": 16,
        "epochs": 15,
        "warmup_ratio": 0.1,
        "label_smoothing": 0.1,
        "gradient_accumulation": 2,
        "lr_scheduler": "cosine_with_restarts"  # ["linear", "cosine", "cosine_with_restarts"]
    }
    
    # Override defaults with passed config
    if config:
        for key, value in config.items():
            default_config[key] = value
    
    config = default_config
    
    # Set model output directory (use parameter if provided, otherwise use default)
    model_output_dir = output_dir if output_dir else MODEL_OUTPUT_DIR
    
    # Apply test mode adjustments
    if test_mode:
        config["learning_rate"] = 2e-4
        config["batch_size"] = 8
        config["epochs"] = 1
        config["gradient_accumulation"] = 1
    
    # Apply simplified model adjustments
    if use_simplified_model:
        print("\n=== Using SIMPLIFIED RoBERTa+CRF Model ===")
        # Adjust config for simplified model
        config["learning_rate"] = 2e-5  # More conservative learning rate
        config["batch_size"] = min(config["batch_size"], 8)  # Smaller batch size
        config["epochs"] = min(config["epochs"], 5)  # Fewer epochs initially
        config["skip_augmentation"] = True  # Skip augmentation for baseline
        config["skip_postprocessing_eval"] = True  # Skip post-processing
    else:
        print("\n=== Using COMPLEX Multi-Loss Model ===")
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load dataset
    dataset = load_from_disk(dataset_path)
    
    if test_mode:
        # Create tiny subset of data for quick testing
        for split in dataset:
            dataset[split] = dataset[split].select(range(min(100, len(dataset[split]))))
        print("TEST MODE: Using small dataset subset")
    
    # Apply data augmentation if not in test mode and not using simplified model
    if not test_mode and not use_simplified_model:
        if config.get("skip_augmentation", False):
            print("Skipping data augmentation for faster optimization")
        else:
            print("Applying data augmentation...")
            dataset["train"] = augment_person_entities(dataset["train"], augmentation_factor=0.3)
    elif use_simplified_model:
        print("Skipping data augmentation for simplified model baseline")
    
    # Load model with tunable parameters
    if test_mode:
        model_name = "distilroberta-base"
    else:
        # Check if the specified model path exists
        if not os.path.exists(model_name) or not any(
            os.path.exists(os.path.join(model_name, f)) 
            for f in ["pytorch_model.bin", "model.safetensors", "config.json"]
        ):
            print(f"Warning: Model path {model_name} not found or incomplete.")
            print("Falling back to base RoBERTa model...")
            model_name = "roberta-base"  # Use base model instead
    
    # Update confidence threshold for post-processing globally
    global confidence_threshold
    confidence_threshold = config["confidence_threshold"]
      # Initialize model based on the flag
    if use_simplified_model:
        print("Initializing SimplifiedRobertaCRFForTokenClassification...")
        model = SimplifiedRobertaCRFForTokenClassification.from_pretrained_custom(
            model_name_or_path=model_name,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            crf_weight=config["crf_weight"]
        )
    else:
        print("Initializing Complex RobertaCRFForTokenClassification...")
        model = RobertaCRFForTokenClassification(
            model_name=model_name,
            num_labels=NUM_LABELS,
            alpha=config["alpha"],
            gamma=config["gamma"],
            person_weight=config["person_weight"],
            crf_weight=config["crf_weight"],
            focal_weight=config["focal_weight"],
            dice_weight=config["dice_weight"],
            # Pass additional parameters for CrossAttentionSpanClassifier
            classifier_params={
                "num_attention_heads": config["num_attention_heads"],
                "max_relative_position": config["max_relative_position"],
                "dropout": config["dropout"]
            },
            # Pass parameters for EnhancedBoundaryDiceLoss
            dice_loss_params={
                "smooth": config["dice_smooth"],
                "b_weight": config["b_weight"],
                "i_end_weight": config["i_end_weight"],
                "context_weight": config["context_weight"]
            }
        )
    
    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # Training arguments with tunable parameters - use model_output_dir
    training_args = TrainingArguments(
        output_dir=model_output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        lr_scheduler_type=config["lr_scheduler"],
        num_train_epochs=config["epochs"],
        logging_dir=os.path.join(LOGS_DIR, "runs"),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="person_f1",
        greater_is_better=True,
        push_to_hub=False,
        label_smoothing_factor=config["label_smoothing"],
        warmup_ratio=config["warmup_ratio"],
        fp16=True,
        report_to="none",
        gradient_accumulation_steps=config["gradient_accumulation"],
        dataloader_pin_memory=False,  # FIX: Disable pin memory to avoid type issues
        remove_unused_columns=False,  # FIX: Keep all columns to avoid issues
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train the model
    print(f"\n=== Starting Training ({'Simplified' if use_simplified_model else 'Complex'} Model) ===")
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate on validation set
    print("\n=== Evaluating on Validation Set ===")
    val_results = trainer.evaluate()
    print(val_results)
    
    # FIX: Save validation results to a standard location
    if output_dir:
        with open(os.path.join(output_dir, "eval_results.json"), 'w') as f:
            json.dump(val_results, f)
    
    # Evaluate on test set
    print("\n=== Evaluating on Test Set ===")
    test_results = trainer.evaluate(dataset["test"])
    print(test_results)
    
    with open(os.path.join(PROJECT_DIR, "logs", "eval_results.json"), 'w') as f:
        json.dump(val_results, f)
    
    # Evaluate post-processing impact only for complex model and not in test mode
    if not test_mode and not use_simplified_model and not config.get("skip_postprocessing_eval", False):
        print("\n=== Evaluating Post-Processing Impact ===")
        try:
            postprocessing_results = evaluate_postprocessing_impact(
                model, 
                dataset["test"], 
                tokenizer
            )
            # Add post-processing improvement to test results
            test_results.update(postprocessing_results)
        except Exception as e:
            print(f"Error during post-processing evaluation: {str(e)}")
            # Continue execution without crashing    elif use_simplified_model:
        print("Skipping post-processing evaluation for simplified model")      # Save final model
    trainer.save_model(model_output_dir)
    
    # Create and save a proper config.json for our custom model
    # This is needed because our custom model doesn't inherit from PreTrainedModel
    import json
    
    # Create custom config that includes all our model parameters
    custom_config = {
        # Base RoBERTa config
        "architectures": ["RobertaCRFForTokenClassification"],  # Our custom architecture
        "model_type": "roberta_crf",  # Custom model type
        
        # Copy important config from underlying RoBERTa model
        "vocab_size": model.roberta.config.vocab_size,
        "hidden_size": model.roberta.config.hidden_size,
        "num_hidden_layers": model.roberta.config.num_hidden_layers,
        "num_attention_heads": model.roberta.config.num_attention_heads,
        "intermediate_size": model.roberta.config.intermediate_size,
        "max_position_embeddings": model.roberta.config.max_position_embeddings,
        "type_vocab_size": model.roberta.config.type_vocab_size,
        "attention_probs_dropout_prob": model.roberta.config.attention_probs_dropout_prob,
        "hidden_dropout_prob": model.roberta.config.hidden_dropout_prob,
        "hidden_act": model.roberta.config.hidden_act,
        "layer_norm_eps": model.roberta.config.layer_norm_eps,
        "initializer_range": model.roberta.config.initializer_range,
        "pad_token_id": model.roberta.config.pad_token_id,
        "bos_token_id": model.roberta.config.bos_token_id,
        "eos_token_id": model.roberta.config.eos_token_id,
        
        # Token classification specific config
        "num_labels": NUM_LABELS,
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
        "problem_type": "token_classification",
        
        # Our custom model parameters
        "alpha": config["alpha"],
        "gamma": config["gamma"],
        "person_weight": config["person_weight"],
        "crf_weight": config["crf_weight"],
        "focal_weight": config["focal_weight"],
        "dice_weight": config["dice_weight"],
        
        # Classifier parameters
        "classifier_params": {
            "num_attention_heads": config["num_attention_heads"],
            "max_relative_position": config["max_relative_position"],
            "dropout": config["dropout"]
        },
        
        # Dice loss parameters
        "dice_loss_params": {
            "smooth": config["dice_smooth"],
            "b_weight": config["b_weight"],
            "i_end_weight": config["i_end_weight"],
            "context_weight": config["context_weight"]
        },
        
        # Other important fields
        "torch_dtype": "float32",
        "transformers_version": "4.36.0",
        "use_cache": True
    }
    
    # Save the custom config
    config_path = os.path.join(model_output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(custom_config, f, indent=2)
    print(f"Saved custom config to {config_path}")
    
    # Save label configuration for consistency during inference
    save_label_config(model_output_dir, include_title_tags)
    
    print(f"Model saved to {model_output_dir}")
    
    # Return results for hyperparameter tuning
    return {
        "person_f1": val_results.get("eval_person_f1", 0.0),
        "entity_f1": val_results.get("eval_entity_f1", 0.0),
        "token_accuracy": val_results.get("eval_token_accuracy", 0.0),
        # Include any other metrics you want to track
        "raw_results": val_results  # Include full results for debugging
    }

