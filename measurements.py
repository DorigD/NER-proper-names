import os
import json
import time
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from datasets import load_from_disk
from tqdm import tqdm
from main import NER
from utils.config import PROJECT_DIR
import warnings
from utils.evaluation_utils import convert_numpy_to_python_types

def count_parameters(model):
    """Count the number of parameters in the model"""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + non_trainable_params
    
    return {
        "trainable": trainable_params,
        "non_trainable": non_trainable_params,
        "total": total_params
    }

def debug_metrics_structure(metrics, dataset_name, model_version):
    """Save the metrics structure to a debug file for inspection"""
    debug_dir = os.path.join(PROJECT_DIR, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Save metrics structure to file
    debug_file = os.path.join(debug_dir, f"metrics_{model_version}_{dataset_name}.json")
    with open(debug_file, "w") as f:
        json.dump(convert_numpy_to_python_types(metrics), f, indent=2)
    
    return debug_file

def extract_entity_metrics(metrics, tag):
    """
    Extract precision, recall, and F1 for a specific entity tag from seqeval metrics
    """
    # For B-PER and I-PER, we need to check for PERSON in the seqeval output
    if tag in ["B-PER", "I-PER"]:
        # Check for PERSON entity metrics
        if "eval_PERSON" in metrics and isinstance(metrics["eval_PERSON"], dict):
            person_metrics = metrics["eval_PERSON"]
            # For B-PER and I-PER, use the same PERSON metrics since seqeval combines them
            return (
                person_metrics.get("precision", 0.0),
                person_metrics.get("recall", 0.0),
                person_metrics.get("f1", 0.0)
            )
        
        # Also try without the eval_ prefix
        if "PERSON" in metrics and isinstance(metrics["PERSON"], dict):
            person_metrics = metrics["PERSON"]
            return (
                person_metrics.get("precision", 0.0),
                person_metrics.get("recall", 0.0),
                person_metrics.get("f1", 0.0)
            )
    
    # Try standard formats - original implementation
    # Check if tag is directly in the metrics dictionary
    if tag in metrics:
        if isinstance(metrics[tag], dict):
            return (
                metrics[tag].get("precision", 0.0),
                metrics[tag].get("recall", 0.0),
                metrics[tag].get("f1", 0.0)
            )
    
    # Try keys with "eval_" prefix (transformers format)
    precision_key = f"eval_{tag}_precision"
    recall_key = f"eval_{tag}_recall"
    f1_key = f"eval_{tag}_f1"
    
    if precision_key in metrics:
        return (
            metrics.get(precision_key, 0.0),
            metrics.get(recall_key, 0.0),
            metrics.get(f1_key, 0.0)
        )
    
    # Try keys without prefix
    precision_key = f"{tag}_precision"
    recall_key = f"{tag}_recall"
    f1_key = f"{tag}_f1"
    
    if precision_key in metrics:
        return (
            metrics.get(precision_key, 0.0),
            metrics.get(recall_key, 0.0),
            metrics.get(f1_key, 0.0)
        )
    
    # Return zeros if nothing is found
    return 0.0, 0.0, 0.0

def evaluate_all_models():
    """
    Evaluate all model versions on all datasets and record performance metrics
    """
    results = []
    
    # Find all model versions (directories starting with "version")
    models_dir = os.path.join(PROJECT_DIR, "models")
    model_versions = [d for d in os.listdir(models_dir) 
                     if d.startswith("version") and os.path.isdir(os.path.join(models_dir, d))]
    print(f"Found {len(model_versions)} model versions")
    
    # Find all datasets
    datasets_dir = os.path.join(PROJECT_DIR, "data", "ds")
    dataset_paths = [os.path.join(datasets_dir, d) for d in os.listdir(datasets_dir) 
                    if os.path.isdir(os.path.join(datasets_dir, d))]
    print(f"Found {len(dataset_paths)} datasets")
    
    # Create NER instance
    ner = NER()
    
    # Evaluate each model on each dataset
    for model_version in tqdm(model_versions, desc="Models"):
        model_path = os.path.join(models_dir, model_version)
        
        try:
            # Measure model loading time
            start_time = time.time()
            ner.load_model(model_path)
            load_time = time.time() - start_time
            
            # Count number of parameters
            model_params = count_parameters(ner.model)
            
            for dataset_path in tqdm(dataset_paths, desc=f"Datasets for {model_version}", leave=False):
                dataset_name = os.path.basename(dataset_path)
                
                try:
                    # Load dataset
                    dataset = load_from_disk(dataset_path)
                    
                    # Determine which split to use (test or validation)
                    if "test" in dataset:
                        test_split = dataset["test"]
                    elif "validation" in dataset:
                        test_split = dataset["validation"]
                    else:
                        # Use first available split if neither test nor validation exists
                        first_split = next(iter(dataset.keys()))
                        test_split = dataset[first_split]
                    
                    # Get dataset size
                    dataset_size = len(test_split)
                    
                    # Measure evaluation time - use the existing evaluate_transformers function
                    start_time = time.time()
                    eval_result = ner.evaluate_transformers(test_split, visualize=False)
                    execution_time = time.time() - start_time
                    
                    if eval_result is None:
                        raise ValueError("Evaluation returned None result")
                    
                    # Extract metrics from the evaluation result
                    metrics = eval_result.get("metrics", {})
                    
                    
                    
                    # Create result record with default values
                    result = {
                        "model_version": model_version,
                        "dataset_name": dataset_name,
                        "model_size_trainable": model_params["trainable"],
                        "model_size_non_trainable": model_params["non_trainable"],
                        "model_size_total": model_params["total"],
                        "dataset_size": dataset_size,
                        "execution_time": execution_time,
                        "load_time": load_time,
                        # Default values for entity metrics
                        "B-PER_precision": 0.0,
                        "B-PER_recall": 0.0,
                        "B-PER_f1": 0.0,
                        "I-PER_precision": 0.0,
                        "I-PER_recall": 0.0,
                        "I-PER_f1": 0.0,
                    }
                    
                    # Extract overall metrics - seqeval usually provides these as 'overall_X'
                    result["overall_precision"] = metrics.get("overall_precision", 
                                               metrics.get("precision", 
                                               metrics.get("eval_precision", 0.0)))
                    
                    result["overall_recall"] = metrics.get("overall_recall", 
                                             metrics.get("recall", 
                                             metrics.get("eval_recall", 0.0)))
                    
                    result["overall_f1"] = metrics.get("overall_f1", 
                                         metrics.get("f1", 
                                         metrics.get("eval_f1", 0.0)))
                    
                    # Extract PER tag metrics 
                    for tag in ["B-PER", "I-PER"]:
                        precision, recall, f1 = extract_entity_metrics(metrics, tag)
                        result[f"{tag}_precision"] = precision
                        result[f"{tag}_recall"] = recall
                        result[f"{tag}_f1"] = f1
                    
                    results.append(result)
                    
                    # Log progress with metrics
                    tqdm.write(f"{model_version} - {dataset_name} (size: {dataset_size}): " +
                              f"PERSON F1={result['B-PER_f1']:.3f}, " +
                              f"P={result['B-PER_precision']:.3f}, " +
                              f"R={result['B-PER_recall']:.3f}, " +
                              f"Time={execution_time:.1f}s")
                    
                except Exception as e:
                    print(f"Error evaluating {model_version} on {dataset_name}: {str(e)}")
                    # Try to get dataset size even if evaluation fails
                    dataset_size = 0
                    try:
                        if 'test_split' in locals():
                            dataset_size = len(test_split)
                    except:
                        pass
                        
                    results.append({
                        "model_version": model_version,
                        "dataset_name": dataset_name,
                        "model_size_trainable": model_params["trainable"],
                        "model_size_non_trainable": model_params["non_trainable"],
                        "model_size_total": model_params["total"],
                        "dataset_size": dataset_size,
                        "execution_time": 0.0,
                        "load_time": load_time,
                        "error": str(e)
                    })
                    
        except Exception as e:
            print(f"Error loading model {model_version}: {str(e)}")
            # Add error entry for all datasets
            for dataset_path in dataset_paths:
                dataset_name = os.path.basename(dataset_path)
                # Try to get dataset size for each dataset
                dataset_size = 0
                try:
                    dataset = load_from_disk(dataset_path)
                    if "test" in dataset:
                        dataset_size = len(dataset["test"])
                    elif "validation" in dataset:
                        dataset_size = len(dataset["validation"])
                    else:
                        first_split = next(iter(dataset.keys()))
                        dataset_size = len(dataset[first_split])
                except:
                    pass
                    
                results.append({
                    "model_version": model_version,
                    "dataset_name": dataset_name,
                    "dataset_size": dataset_size,
                    "error": f"Model loading error: {str(e)}"
                })
        
        # Clean GPU memory after each model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results

def save_results(results):
    """
    Save evaluation results to CSV format only
    """
    results_dir = os.path.join(PROJECT_DIR, "evaluation_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save with readable filename
    csv_path = os.path.join(results_dir, f"model_evaluations.csv")
    
    # Convert to DataFrame for easier CSV handling and analysis
    df = pd.DataFrame(results)
    
    # Save as CSV
    df.to_csv(csv_path, index=False)
    
    print(f"Results saved to:")
    print(f"- CSV: {csv_path}")
    
    # Generate summary statistics
    print("\nSummary Statistics:")
    if "error" not in df.columns or df["error"].count() < len(df):
        try:
            # Summary of F1 scores by model version
            f1_by_model = df.groupby("model_version")["B-PER_f1"].mean().sort_values(ascending=False)
            print(f"\nAverage B-PER F1 score by model version:")
            for model, f1 in f1_by_model.items():
                print(f"{model}: {f1:.4f}")
            
            # Best performing model on each dataset
            best_by_dataset = df.loc[df.groupby("dataset_name")["B-PER_f1"].idxmax()]
            print(f"\nBest model for each dataset:")
            for _, row in best_by_dataset.iterrows():
                print(f"{row['dataset_name']} (size: {row['dataset_size']}): {row['model_version']} (F1={row['B-PER_f1']:.4f})")
            
            # Overall best model
            best_overall = df.loc[df["B-PER_f1"].idxmax()]
            print(f"\nOverall best model: {best_overall['model_version']} on {best_overall['dataset_name']} (F1={best_overall['B-PER_f1']:.4f})")
        except Exception as e:
            print(f"Could not generate summary statistics: {e}")
    
    return csv_path

def main():
    print("Starting model evaluation across all versions and datasets")
    # Suppress warnings during evaluation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = evaluate_all_models()
    save_results(results)
    print("Evaluation complete!")

if __name__ == "__main__":
    main()