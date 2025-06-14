"""
Person-Only NER Model Evaluation Pipeline with Timing and Size Metrics
=====================================================================

This script evaluates RoBERTa-based Hugging Face NER models specifically on person entity recognition,
using preprocessed HuggingFace datasets. It calculates F1, precision, and recall metrics for person entities
(B-PERSON/I-PERSON in dataset mapped to B-PER/I-PER in models) while ignoring O tags.

Features:
- Tests RoBERTa-based pre-trained HuggingFace NER models
- Uses preprocessed HuggingFace datasets created via preprocess.py
- Entity-level evaluation (combines B-PERSON/I-PERSON into complete person entities)
- Token-level evaluation for person tags
- Model size and inference time tracking
- Batch evaluation across multiple datasets
- Comprehensive performance comparison

Usage:
    python person_ner_evaluation_enhanced.py
"""

import os
import json
import pandas as pd
import sys
import time
from datetime import datetime
from typing import List, Dict, Tuple, Any
import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, 
    pipeline
)
from datasets import Dataset, DatasetDict, load_from_disk
from sklearn.metrics import precision_recall_fscore_support, classification_report

# Add utils to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import DATA_DIR

# Configuration - RoBERTa-based models
ROBERTA_MODELS = [
    "Jean-Baptiste/roberta-large-ner-english",
    "xlm-roberta-large-finetuned-conll03-english"
]

# Dataset label mapping (your dataset -> model expected)
DATASET_LABEL_MAPPING = {
    0: "O",        # O -> O
    1: "B-PER",    # B-PERSON -> B-PER
    2: "I-PER"     # I-PERSON -> I-PER
}

# Reverse mapping for model outputs
MODEL_PERSON_LABELS = ["PER", "PERSON"]  # Different models might use different labels

def calculate_model_size(model) -> tuple:
    """
    Calculate model size in megabytes and total parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    total_size = sum(p.numel() * p.element_size() for p in model.parameters())
    size_mb = total_size / (1024 * 1024)
    return size_mb, total_params

def load_dataset_from_path(dataset_path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Load dataset from various sources and return tokens and labels.
    """
    print(f"Loading dataset from {dataset_path}")
    
    # First, try to load raw test data if available
    raw_test_path = os.path.join(dataset_path, "raw_test_data.json")
    if os.path.exists(raw_test_path):
        print("  Using raw test data")
        with open(raw_test_path, 'r') as f:
            raw_data = json.load(f)
        
        sentences = []
        labels = []
        
        for item in raw_data:
            tokens = item["tokens"]
            # Convert numeric labels to string labels
            label_ints = item["labels"]
            label_strings = [DATASET_LABEL_MAPPING.get(label_id, "O") for label_id in label_ints]
            
            sentences.append(tokens)
            labels.append(label_strings)
            
        print(f"  Loaded {len(sentences)} sentences from raw test data")
        return sentences, labels
    
    # Try HuggingFace dataset
    try:
        dataset_dict = load_from_disk(dataset_path)
        print(f"  Dataset splits available: {list(dataset_dict.keys())}")
        
        # Use test split for evaluation, fallback to train
        if "test" in dataset_dict:
            test_dataset = dataset_dict["test"]
            print("  Using test split")
        elif "train" in dataset_dict:
            print("  No test split found, using train split")
            test_dataset = dataset_dict["train"]
        else:
            # Handle single dataset case
            print("  Loading single dataset")
            test_dataset = dataset_dict
        
        # Load tokenizer for decoding
        metadata_path = os.path.join(dataset_path, "dataset_metadata.json")
        tokenizer_name = "roberta-base"  # default
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            tokenizer_name = metadata.get('model_name', 'roberta-base')
        
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=True)
        
        sentences = []
        labels = []
        
        # Limit to reasonable size for evaluation
        max_samples = min(500, len(test_dataset))
        print(f"  Processing {max_samples} samples")
        
        for i in range(max_samples):
            example = test_dataset[i]
            
            # Decode tokens
            input_ids = example['input_ids']
            label_ids = example['labels']
            
            # Remove special tokens and padding
            valid_indices = [j for j, (input_id, label_id) in enumerate(zip(input_ids, label_ids)) 
                           if label_id != -100 and input_id not in [tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id]]
            
            if not valid_indices:
                continue
                
            valid_input_ids = [input_ids[j] for j in valid_indices]
            valid_label_ids = [label_ids[j] for j in valid_indices]
            
            # Decode to tokens
            tokens = [tokenizer.decode([input_id]).strip() for input_id in valid_input_ids]
            
            # Convert label IDs to label strings
            label_strings = [DATASET_LABEL_MAPPING.get(label_id, "O") for label_id in valid_label_ids]
            
            if tokens and len(tokens) == len(label_strings):
                sentences.append(tokens)
                labels.append(label_strings)
        
        print(f"  Successfully reconstructed {len(sentences)} sentences")
        return sentences, labels
        
    except Exception as e:
        print(f"  Error loading HuggingFace dataset: {e}")
        raise

def filter_person_entities(sentences: List[List[str]], labels: List[List[str]]) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Filter sentences to only include those with person entities.
    """
    filtered_sentences = []
    filtered_labels = []
    
    for sent, lbls in zip(sentences, labels):
        # Keep sentences that have at least one person tag (B-PER or I-PER)
        if any(label in ['B-PER', 'I-PER'] for label in lbls):
            filtered_sentences.append(sent)
            filtered_labels.append(lbls)
    
    return filtered_sentences, filtered_labels

def extract_person_entities(tokens: List[str], labels: List[str]) -> List[Tuple[int, int, str]]:
    """
    Extract person entities from BIO labels.
    """
    entities = []
    current_entity_start = None
    current_entity_tokens = []
    
    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label == 'B-PER':
            # Start new entity
            if current_entity_start is not None:
                # End previous entity
                entity_text = ' '.join(current_entity_tokens)
                entities.append((current_entity_start, i-1, entity_text))
            current_entity_start = i
            current_entity_tokens = [token]
        elif label == 'I-PER' and current_entity_start is not None:
            # Continue current entity
            current_entity_tokens.append(token)
        else:
            # End current entity if exists
            if current_entity_start is not None:
                entity_text = ' '.join(current_entity_tokens)
                entities.append((current_entity_start, i-1, entity_text))
                current_entity_start = None
                current_entity_tokens = []
    
    # Handle entity at end of sentence
    if current_entity_start is not None:
        entity_text = ' '.join(current_entity_tokens)
        entities.append((current_entity_start, len(tokens)-1, entity_text))
    
    return entities

def evaluate_model_on_dataset(model_name: str, dataset_path: str) -> Dict[str, Any]:
    """
    Evaluate a model on a single dataset with comprehensive metrics.
    """
    print(f"\n Evaluating {model_name} on {os.path.basename(dataset_path)}")
    
    try:
        # Load dataset
        test_sentences, test_labels = load_dataset_from_path(dataset_path)
        
        # Filter to person entities only
        test_sentences, test_labels = filter_person_entities(test_sentences, test_labels)
        print(f"  Sentences with person entities: {len(test_sentences)}")
        
        if len(test_sentences) == 0:
            return {
                'error': 'No sentences with person entities found',
                'dataset_path': dataset_path,
                'model_name': model_name
            }
        
        # Limit for faster evaluation
        max_test_size = 2000000
        if len(test_sentences) > max_test_size:
            test_sentences = test_sentences[:max_test_size]
            test_labels = test_labels[:max_test_size]
        
        # Load model and tokenizer with timing
        model_load_start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        model_load_time = time.time() - model_load_start
        
        # Calculate model size
        model_size_mb, total_params = calculate_model_size(model)
        print(f"    Model: {model_size_mb:.1f} MB ({total_params:,} params)")
        
        # Create pipeline with timing
        pipeline_start = time.time()
        ner_pipeline = pipeline(
            "ner", 
            model=model, 
            tokenizer=tokenizer, 
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1
        )
        pipeline_setup_time = time.time() - pipeline_start
        
        # Track inference metrics
        total_inference_time = 0
        inference_times = []
        predicted_entities_all = []
        true_entities_all = []
        predicted_labels_token = []
        true_labels_token = []
        
        print(f"    Processing {len(test_sentences)} sentences...")
        
        for i, (sentence, true_labels) in enumerate(zip(test_sentences, test_labels)):
            text = ' '.join(sentence)
            
            if len(text.strip()) < 3:
                continue
            
            try:
                # Time the inference
                inference_start = time.time()
                predictions = ner_pipeline(text[:2000])  # Truncate if too long
                inference_time = time.time() - inference_start
                
                total_inference_time += inference_time
                inference_times.append(inference_time)
                
                # Initialize predictions
                predicted_bio = ['O'] * len(sentence)
                
                # Map predictions to tokens
                for pred in predictions:
                    entity_label = pred['entity_group'].upper()
                    if any(person_label in entity_label for person_label in MODEL_PERSON_LABELS):
                        start_char = pred['start']
                        end_char = pred['end']
                        
                        char_pos = 0
                        first_token = True
                        
                        for token_idx, token in enumerate(sentence):
                            token_start = char_pos
                            token_end = char_pos + len(token)
                            
                            if token_start < end_char and token_end > start_char:
                                if predicted_bio[token_idx] == 'O':
                                    predicted_bio[token_idx] = 'B-PER' if first_token else 'I-PER'
                                    first_token = False
                            
                            char_pos = token_end + 1
                
                # Extract entities
                true_entities = extract_person_entities(sentence, true_labels)
                predicted_entities = extract_person_entities(sentence, predicted_bio)
                
                true_entities_all.extend([(i, start, end, text) for start, end, text in true_entities])
                predicted_entities_all.extend([(i, start, end, text) for start, end, text in predicted_entities])
                
                # Collect token-level labels
                for true_label, pred_label in zip(true_labels, predicted_bio):
                    if true_label in ['B-PER', 'I-PER']:
                        true_labels_token.append(true_label)
                        predicted_labels_token.append(pred_label if pred_label in ['B-PER', 'I-PER'] else 'O')
                        
            except Exception as e:
                print(f"    Error processing sentence {i}: {e}")
                continue
        
        # Calculate metrics
        true_entities_set = set(true_entities_all)
        predicted_entities_set = set(predicted_entities_all)
        
        entity_tp = len(true_entities_set & predicted_entities_set)
        entity_fp = len(predicted_entities_set - true_entities_set)
        entity_fn = len(true_entities_set - predicted_entities_set)
        
        entity_precision = entity_tp / (entity_tp + entity_fp) if (entity_tp + entity_fp) > 0 else 0
        entity_recall = entity_tp / (entity_tp + entity_fn) if (entity_tp + entity_fn) > 0 else 0
        entity_f1 = 2 * entity_precision * entity_recall / (entity_precision + entity_recall) if (entity_precision + entity_recall) > 0 else 0
        
        # Token-level metrics
        token_precision = token_recall = token_f1 = 0
        if len(true_labels_token) > 0:
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels_token, predicted_labels_token, 
                labels=['B-PER', 'I-PER'], average='weighted', zero_division=0
            )
            token_precision, token_recall, token_f1 = precision, recall, f1
        
        # Timing statistics
        avg_inference_time = total_inference_time / len(inference_times) if inference_times else 0
        
        print(f"    Results: Entity F1={entity_f1:.3f}, Token F1={token_f1:.3f}, {avg_inference_time*1000:.1f}ms avg")
        
        return {
            'dataset_name': os.path.basename(dataset_path),
            'dataset_path': dataset_path,
            'model_name': model_name,
            'model_size_mb': model_size_mb,
            'total_parameters': total_params,
            'model_load_time_seconds': model_load_time,
            'pipeline_setup_time_seconds': pipeline_setup_time,
            'total_inference_time_seconds': total_inference_time,
            'avg_inference_time_seconds': avg_inference_time,
            'avg_inference_time_ms': avg_inference_time * 1000,
            'sentences_processed': len(inference_times),
            'sentences_per_second': len(inference_times) / total_inference_time if total_inference_time > 0 else 0,
            'entity_precision': entity_precision,
            'entity_recall': entity_recall,
            'entity_f1': entity_f1,
            'token_precision': token_precision,
            'token_recall': token_recall,
            'token_f1': token_f1,
            'true_entities': len(true_entities_set),
            'predicted_entities': len(predicted_entities_set),
            'person_tokens': len(true_labels_token),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"     Error: {e}")
        return {
            'dataset_name': os.path.basename(dataset_path),
            'dataset_path': dataset_path,
            'model_name': model_name,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def find_no_title_datasets() -> List[str]:
    """
    Find all NO-TITLE datasets.
    """
    no_title_path = os.path.join(DATA_DIR, "ds", "NO-TITLE")
    datasets = []
    
    if os.path.exists(no_title_path):
        for item in os.listdir(no_title_path):
            dataset_path = os.path.join(no_title_path, item)
            if os.path.isdir(dataset_path):
                # Check if it's a valid dataset
                if (os.path.exists(os.path.join(dataset_path, "dataset_info.json")) or
                    os.path.exists(os.path.join(dataset_path, "raw_test_data.json"))):
                    datasets.append(dataset_path)
    
    return datasets

def run_batch_evaluation():
    """
    Run evaluation on all NO-TITLE datasets with all RoBERTa models.
    """
    print("Person NER Evaluation with Timing and Size Metrics")
    print("=" * 60)
    
    # Find datasets
    datasets = find_no_title_datasets()
    
    if not datasets:
        print(" No NO-TITLE datasets found!")
        return
    
    print(f"Found {len(datasets)} NO-TITLE datasets:")
    for dataset in datasets:
        print(f"  • {os.path.basename(dataset)}")
    
    print(f"\nTesting {len(ROBERTA_MODELS)} RoBERTa models:")
    for model in ROBERTA_MODELS:
        print(f"  • {model}")
    
    # Run evaluations
    all_results = []
    total_combinations = len(datasets) * len(ROBERTA_MODELS)
    current_combination = 0
    
    for dataset_path in datasets:
        for model_name in ROBERTA_MODELS:
            current_combination += 1
            print(f"\n[{current_combination}/{total_combinations}] Evaluating {model_name} on {os.path.basename(dataset_path)}")
            
            result = evaluate_model_on_dataset(model_name, dataset_path)
            all_results.append(result)
      # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "person_evaluation_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save all results to JSON for detailed analysis
    results_file = os.path.join(results_dir, f"detailed_base_model_person_ner_evaluation_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Create comprehensive DataFrame
    valid_results = [r for r in all_results if 'error' not in r]
    error_results = [r for r in all_results if 'error' in r]
    
    if valid_results:
        # Detailed results CSV
        df_detailed = pd.DataFrame(valid_results)
        detailed_csv = os.path.join(results_dir, f"detailed_person_ner_results_{timestamp}.csv")
        df_detailed.to_csv(detailed_csv, index=False)
          # Create summary CSV with key metrics (matching existing CSV structure)
        summary_data = []
        for result in valid_results:
            # Determine model type based on model name
            if "Jean-Baptiste/roberta-large-ner-english" in result['model_name']:
                model_type = "roberta-large-ner-english"
            elif "xlm-roberta-large-finetuned-conll03-english" in result['model_name']:
                model_type = "xlm-roberta-large-finetuned-conll03-english"
            else:
                model_type = result['model_name'].split('/')[-1]  # Fallback to short name            
            summary_row = {
                'dataset': result['dataset_name'],
                'model_type': model_type,  # Use model type instead of full model name
                'num_samples': result.get('sentences_processed', 0),
                'inference_time_seconds': result.get('total_inference_time_seconds', 0),
                'inference_time_per_sample': result['avg_inference_time_seconds'],
                'person_precision': result['entity_precision'],
                'person_recall': result['entity_recall'],
                'person_f1': result['entity_f1'],
                'entity_f1_macro': result['entity_f1'],  # Use entity_f1 as macro average
                'token_accuracy': result['token_f1'],  # Use token_f1 as accuracy metric
                'timestamp': result.get('timestamp', datetime.now().isoformat()),
                'model_size_mb': result['model_size_mb'],
                'sentences_per_second': result.get('sentences_per_second', 0)  # Add missing field
            }
            summary_data.append(summary_row)        
        df_summary = pd.DataFrame(summary_data)
        summary_csv = os.path.join(results_dir, f"detailed_base_model_person_ner_evaluation_{timestamp}.csv")
        df_summary.to_csv(summary_csv, index=False)
        
        # Create aggregated results by model (like notebook)
        aggregated_data = []
        for model_name in df_summary['model_type'].unique():
            model_data = df_summary[df_summary['model_type'] == model_name]
            
            agg_row = {
                'model_type': model_name,
                'num_datasets': len(model_data),
                'total_samples': model_data['num_samples'].sum(),
                'total_inference_time': model_data['inference_time_seconds'].sum(),
                'avg_inference_time_per_sample': model_data['inference_time_per_sample'].mean(),
                'model_size_mb': model_data['model_size_mb'].iloc[0],
                
                # Person metrics
                'person_f1_mean': model_data['person_f1'].mean(),
                'person_f1_std': model_data['person_f1'].std() if len(model_data) > 1 else 0,
                'person_f1_min': model_data['person_f1'].min(),
                'person_f1_max': model_data['person_f1'].max(),
                
                'person_precision_mean': model_data['person_precision'].mean(),
                'person_precision_std': model_data['person_precision'].std() if len(model_data) > 1 else 0,
                
                'person_recall_mean': model_data['person_recall'].mean(),
                'person_recall_std': model_data['person_recall'].std() if len(model_data) > 1 else 0,
                
                'token_accuracy_mean': model_data['token_accuracy'].mean(),
                'token_accuracy_std': model_data['token_accuracy'].std() if len(model_data) > 1 else 0,
                
                'avg_sentences_per_second': model_data['sentences_per_second'].mean(),
                'timestamp': datetime.now().isoformat()
            }
            aggregated_data.append(agg_row)
        
        df_aggregated = pd.DataFrame(aggregated_data)
        aggregated_csv = os.path.join(results_dir, f"aggregated_person_ner_results_{timestamp}.csv")
        df_aggregated.to_csv(aggregated_csv, index=False)
        
        # Print results
        print("\n" + "="*120)
        print("PERSON NER EVALUATION RESULTS (DETAILED)")
        print("="*120)
        
        display_cols = ['dataset', 'model_type', 'person_f1', 'person_precision', 'person_recall', 
                       'token_accuracy', 'model_size_mb', 'sentences_per_second']
        print(df_summary[display_cols].round(4).to_string(index=False))
        
        # Aggregated summary
        if len(df_aggregated) > 0:
            print("\n" + "="*120)
            print("AGGREGATED MODEL PERFORMANCE SUMMARY")
            print("="*120)
            
            agg_display_cols = ['model_type', 'num_datasets', 'person_f1_mean', 'person_f1_std',
                               'person_precision_mean', 'person_recall_mean', 'model_size_mb', 
                               'avg_inference_time_per_sample', 'avg_sentences_per_second']
            
            print(df_aggregated[agg_display_cols].round(4).to_string(index=False))
            
            # Model comparison
            if len(df_aggregated) >= 2:
                print(f"\n MODEL COMPARISON:")
                best_f1_model = df_aggregated.loc[df_aggregated['person_f1_mean'].idxmax()]
                fastest_model = df_aggregated.loc[df_aggregated['avg_sentences_per_second'].idxmax()]
                smallest_model = df_aggregated.loc[df_aggregated['model_size_mb'].idxmin()]
                
                print(f"  Best F1: {best_f1_model['model_type']} ({best_f1_model['person_f1_mean']:.4f})")
                print(f"  Fastest: {fastest_model['model_type']} ({fastest_model['avg_sentences_per_second']:.1f} sent/sec)")
                print(f"  Smallest: {smallest_model['model_type']} ({smallest_model['model_size_mb']:.1f} MB)")
          # Save summary report
        summary_report = os.path.join(results_dir, f"evaluation_summary_report_{timestamp}.txt")
        with open(summary_report, 'w') as f:
            f.write("Person NER Evaluation Summary Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Evaluations: {len(valid_results)}\n")
            f.write(f"Failed Evaluations: {len(error_results)}\n\n")
            
            f.write("Models Evaluated:\n")
            for model in df_summary['model_type'].unique():
                f.write(f"  - {model}\n")
            
            f.write("\nDatasets Evaluated:\n")
            for dataset in df_summary['dataset'].unique():
                f.write(f"  - {dataset}\n")
            
            f.write("\nAggregated Results:\n")
            f.write(df_aggregated[agg_display_cols].round(4).to_string(index=False))
            
            if error_results:
                f.write("\n\nErrors:\n")
                for error in error_results:
                    f.write(f"  - {error['dataset_name']} with {error['model_name']}: {error['error']}\n")
        
        print(f"\n FILES SAVED:")
        print(f"   Detailed results: {detailed_csv}")
        print(f"   Summary results: {summary_csv}")
        print(f"   Aggregated results: {aggregated_csv}")
        print(f"  📝 Summary report: {summary_report}")
        print(f"  🗃️  JSON backup: {results_file}")
        
    else:
        print(" No successful evaluations to save!")
        
    if error_results:
        print(f"\n  {len(error_results)} evaluations failed:")
        for error in error_results:
            print(f"  • {error['dataset_name']} + {error['model_name']}: {error['error'][:100]}...")
    
    print(f"\n Batch evaluation completed!")

if __name__ == "__main__":
    run_batch_evaluation()
