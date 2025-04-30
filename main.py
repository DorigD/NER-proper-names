# First, import torch completely before anything else
import torch
import numpy as np
import os
import json

# Then import transformers
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Finally import your local modules
from utils.config import PROJECT_DIR
from scripts.train import train_model
from scripts.preprocess import preprocess
from scripts.converters.Converter import convert_file
class NER:
    def __init__(self):
        """
        Initialize the NER (Named Entity Recognition) class.
        This can include loading models, setting up configurations, etc.
        """
        self.base_model = "roberta-base"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.id2label = None

    def load_model(self, model_path=None):
        """
        Load the model and tokenizer.
        
        :param model_path: Optional path to the model. If None, uses the default path.
        """
        if model_path is None:
            model_path = os.path.join(PROJECT_DIR, "models", "roberta-finetuned-ner")
        
        # Load model config to get id2label mapping
        with open(os.path.join(model_path, "model_config.json"), "r") as f:
            model_config = json.load(f)
            
        self.id2label = model_config["id2label"]
        self.model = AutoModelForTokenClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
        return self

    def train(self, training_data=None):
        """
        Train the NER model using the provided training data.

        :param training_data: List of training examples.
        """
        try:
            # Preprocess the training data
            if training_data is None or os.path.isdir(training_data):
                training_data = os.path.join(PROJECT_DIR, "data", "raw")
                for file in os.listdir(training_data):
                    if file.endswith(".txt") or file.endswith(".csv"):
                        convert_file(os.path.join(training_data, file), replace=False)
                        break
            else:
                convert_file(training_data)
        except Exception as e:
            print(f"Error during data conversion: {e}")
            return
        try:
            # Preprocess the training data
            preprocess()
        except Exception as e:
            print(f"Error during data preprocessing: {e}")
            return
        train_model(self.base_model)

    def set_model(self, model_path):
        """
        Set the model for the NER class.

        :param model_path: Path to the pre-trained model.
        """
        self.base_model = model_path

    def predict(self, text):
        """
        Predict named entities in the given text.

        :param text: Input string to analyze.
        :return: List of named entities found in the text.
        """
        # Load model if not already loaded
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        # Tokenize input text
        encoded_input = self.tokenizer(text, 
                                    return_tensors="pt", 
                                    padding=True, 
                                    truncation=True, 
                                    return_offsets_mapping=True)
        
        # Get token offsets for mapping predictions back to original text
        offset_mapping = encoded_input.pop("offset_mapping")
        
        # Move tensors to device
        input_ids = encoded_input["input_ids"].to(self.device)
        attention_mask = encoded_input["attention_mask"].to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get predictions
        predictions = np.argmax(outputs.logits.cpu().numpy(), axis=2)[0]
        
        # Map predictions to entities in the original text
        entities = []
        current_entity = None
        
        # Skip special tokens ([CLS], [SEP], [PAD])
        for idx, (pred, offset) in enumerate(zip(predictions[1:], offset_mapping[0][1:])):
            # Skip if this is a special token or padding (offset will be (0,0))
            if offset[0] == offset[1]:
                continue
                
            # Get the predicted label
            label = self.id2label[str(pred)]
            
            # Handle entity boundaries
            if label.startswith("B-"):
                # If we had a previous entity, add it to our results
                if current_entity:
                    entities.append(current_entity)
                    
                # Start a new entity
                entity_type = label[2:]  # Remove "B-" prefix
                current_entity = {
                    "type": entity_type,
                    "start": int(offset[0]),
                    "end": int(offset[1]),
                    "text": text[int(offset[0]):int(offset[1])]
                }
            
            elif label.startswith("I-") and current_entity:
                # Continue the current entity if the I- tag matches the current entity type
                if label[2:] == current_entity["type"]:
                    current_entity["end"] = int(offset[1])
                    current_entity["text"] = text[current_entity["start"]:current_entity["end"]]
                    
            elif label == "O" and current_entity:
                # End of an entity
                entities.append(current_entity)
                current_entity = None
        
        # Add the last entity if there is one
        if current_entity:
            entities.append(current_entity)
        
        return entities

    def predict_batch(self, texts):
        """
        Predict named entities for multiple texts efficiently.
        
        :param texts: List of strings to analyze
        :return: List of entity lists, one per input text
        """
        # Load model if not already loaded
        if self.model is None or self.tokenizer is None:
            self.load_model()
            
        # Process all texts in a single batch
        batch_results = []
        
        # Tokenize in batch
        encoded_inputs = self.tokenizer(texts, 
                                       return_tensors="pt", 
                                       padding=True, 
                                       truncation=True, 
                                       return_offsets_mapping=True)
        
        # Get token offsets for mapping predictions back to original text
        offset_mappings = encoded_inputs.pop("offset_mapping")
        
        # Move tensors to device
        input_ids = encoded_inputs["input_ids"].to(self.device)
        attention_mask = encoded_inputs["attention_mask"].to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get predictions
        predictions = np.argmax(outputs.logits.cpu().numpy(), axis=2)
        
        # Process each text's predictions
        for i, (text, preds, offsets) in enumerate(zip(texts, predictions, offset_mappings)):
            entities = []
            current_entity = None
            
            # Skip special tokens ([CLS], [SEP], [PAD])
            for idx, (pred, offset) in enumerate(zip(preds[1:], offsets[1:])):
                # Skip if this is a special token or padding
                if offset[0] == offset[1]:
                    continue
                    
                # Get the predicted label
                label = self.id2label[str(pred)]
                
                # Handle entity boundaries (similar logic to predict method)
                if label.startswith("B-"):
                    if current_entity:
                        entities.append(current_entity)
                        
                    entity_type = label[2:]
                    current_entity = {
                        "type": entity_type,
                        "start": int(offset[0]),
                        "end": int(offset[1]),
                        "text": text[int(offset[0]):int(offset[1])]
                    }
                
                elif label.startswith("I-") and current_entity:
                    if label[2:] == current_entity["type"]:
                        current_entity["end"] = int(offset[1])
                        current_entity["text"] = text[current_entity["start"]:current_entity["end"]]
                        
                elif label == "O" and current_entity:
                    entities.append(current_entity)
                    current_entity = None
            
            # Add the last entity if there is one
            if current_entity:
                entities.append(current_entity)
            
            batch_results.append(entities)
        
        return batch_results
    
    
    def evaluate(self, test_dataset=None, visualize=False):
        """
        Evaluate model performance on a test dataset with comprehensive metrics
        
        Args:
            test_dataset: HuggingFace dataset for testing. If None, will load from default path.
            visualize: Whether to generate and display visualizations (default: False)
            
        Returns:
            Dictionary of evaluation metrics and analysis results
        """
        from datasets import load_from_disk
        from sklearn.metrics import classification_report, confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        from utils.metrics import extract_entities, compute_metrics
        from utils.evaluation_utils import (
            analyze_errors, 
            visualize_entity_performance, 
            display_context_examples, 
            analyze_confidence,
            convert_numpy_to_python_types
        )
        from utils.config import DATASET_PATH, MODELS_DIR, PROJECT_DIR
        import os
        from datetime import datetime
        
        # Load model if not already loaded
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        # Create results directory
        results_dir = os.path.join(PROJECT_DIR, "evaluation_results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Load test dataset if not provided
        if test_dataset is None:
            test_dataset = load_from_disk(DATASET_PATH)["test"]
        
        # Get model name for results
        model_path = os.path.join(MODELS_DIR, "roberta-finetuned-ner")
        model_name = os.path.basename(model_path)
        
        # Run predictions
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        for batch in test_dataset:
            input_ids = torch.tensor([batch["input_ids"]]).to(self.device)
            attention_mask = torch.tensor([batch["attention_mask"]]).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            predictions = np.argmax(outputs.logits.cpu().numpy(), axis=2)[0]
            label_ids = batch["labels"]
            
            # Filter out padding tokens (-100)
            true_predictions = [p for p, l in zip(predictions, label_ids) if l != -100]
            true_labels = [l for l in label_ids if l != -100]
            
            all_predictions.extend(true_predictions)
            all_labels.extend(true_labels)
        
        # Generate detailed metrics
        report = classification_report(
            all_labels, 
            all_predictions, 
            labels=list(range(len(self.id2label))),
            target_names=list(self.id2label.values()),
            digits=4
        )
        
        if visualize:
            print(f"Model: {model_name}")
            print("Classification Report:")
            print(report)
        
        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        if visualize:
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=list(self.id2label.values()),
                yticklabels=list(self.id2label.values())
            )
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'confusion_matrix_{model_name}.png'))
            plt.show()
        
        # Calculate entity-level metrics
        true_entities = extract_entities(all_labels)
        pred_entities = extract_entities(all_predictions)
        
        # Calculate metrics directly
        correct = len(set(true_entities) & set(pred_entities))
        precision = correct / len(pred_entities) if pred_entities else 0
        recall = correct / len(true_entities) if true_entities else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        entity_metrics = {"precision": precision, "recall": recall, "f1": f1}
        
        if visualize:
            print("\nEntity-level metrics:")
            print(f"Precision: {entity_metrics['precision']:.4f}")
            print(f"Recall: {entity_metrics['recall']:.4f}")
            print(f"F1 Score: {entity_metrics['f1']:.4f}")
            print(f"Matched entities: {correct} out of {len(true_entities)} true and {len(pred_entities)} predicted")
        
        # Perform error analysis
        error_analysis = analyze_errors(
            self.id2label, all_labels, all_predictions, test_dataset, 
            self.tokenizer, visualize
        )
        
        if visualize:
            visualize_entity_performance(true_entities, pred_entities)
            display_context_examples(self.tokenizer, test_dataset, all_labels, all_predictions)
            confidence_stats = analyze_confidence(self.model, self.device, test_dataset, all_labels, all_predictions)
        else:
            confidence_stats = {}
        
        # Save evaluation results
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        results = {
            "model_name": model_name,
            "timestamp": timestamp,
            "classification_report": report,
            "entity_metrics": entity_metrics,
            "error_analysis": error_analysis,
            "confidence_stats": confidence_stats if visualize else {}
        }
        with open(os.path.join(results_dir, f"evaluation_{model_name}_{timestamp}.json"), "w") as f:
            # Convert NumPy types before serialization
            serializable_results = convert_numpy_to_python_types(results)
            json.dump(serializable_results, f, indent=2)
        if visualize:
            print(f"Evaluation results saved to {os.path.join(results_dir, f'evaluation_{model_name}_{timestamp}.json')}")
        
        return results

    def evaluate_compare(self, model_paths, visualize=False):
        """
        Compare this model with other models
        
        Args:
            model_paths: List of model directory paths to compare against
            visualize: Whether to generate and display visualizations
            
        Returns:
            DataFrame with comparison metrics
        """
        import pandas as pd
        from datasets import load_from_disk
        from utils.config import DATASET_PATH
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        from datetime import datetime
        
        # Load test dataset
        test_dataset = load_from_disk(DATASET_PATH)["test"]
        
        # Create results directory
        results_dir = os.path.join(PROJECT_DIR, "evaluation_results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Evaluate current model
        current_results = self.evaluate(test_dataset, visualize=False)
        
        # Store comparison results
        results = [{
            "Model": "Current Model",
            "Precision": current_results["entity_metrics"]["precision"],
            "Recall": current_results["entity_metrics"]["recall"],
            "F1 Score": current_results["entity_metrics"]["f1"]
        }]
        
        # Evaluate each model in model_paths
        for model_path in model_paths:
            model_name = os.path.basename(model_path)
            if visualize:
                print(f"\n\n{'='*50}\nEvaluating model: {model_name}\n{'='*50}\n")
            
            temp_ner = NER()
            temp_ner.load_model(model_path)
            model_results = temp_ner.evaluate(test_dataset, visualize=visualize)
            
            results.append({
                "Model": model_name,
                "Precision": model_results["entity_metrics"]["precision"],
                "Recall": model_results["entity_metrics"]["recall"],
                "F1 Score": model_results["entity_metrics"]["f1"]
            })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        if visualize:
            # Print comparison table
            print("\n\nModel Comparison Summary:")
            print(df.to_string(index=False, float_format="%.4f"))
            
            # Create bar chart comparison
            metrics = ["Precision", "Recall", "F1 Score"]
            models = df["Model"].tolist()
            
            plt.figure(figsize=(12, 8))
            x = np.arange(len(models))
            width = 0.25
            
            for i, metric in enumerate(metrics):
                values = df[metric].tolist()
                plt.bar(x + (i-1)*width, values, width, label=metric)
            
            plt.ylabel('Score')
            plt.title('Model Performance Comparison')
            plt.xticks(x, models, rotation=45)
            plt.ylim(0, 1.05)
            plt.legend()
            plt.tight_layout()
            
            # Save comparison
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            plt.savefig(os.path.join(results_dir, f'model_comparison_{timestamp}.png'))
            plt.show()
            
            # Save comparison data
            df.to_csv(os.path.join(results_dir, f'model_comparison_{timestamp}.csv'), index=False)
            print(f"Comparison saved to {os.path.join(results_dir, f'model_comparison_{timestamp}.csv')}")
        
        return df

        """Helper method to analyze model confidence"""
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        from utils.config import PROJECT_DIR
        
        results_dir = os.path.join(PROJECT_DIR, "evaluation_results")
        
        all_confidences = []
        all_correctness = []
        
        token_idx = 0
        
        for batch in test_dataset:
            input_ids = torch.tensor([batch["input_ids"]]).to(self.device)
            attention_mask = torch.tensor([batch["attention_mask"]]).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Get softmax probabilities
            logits = outputs.logits.cpu().numpy()[0]
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
            
            # Get confidence of selected class
            predictions = np.argmax(logits, axis=1)
            confidences = [probs[i, p] for i, p in enumerate(predictions)]
            
            # Match with labels
            labels = batch["labels"]
            
            # Process non-padding tokens
            for i, (conf, pred, label) in enumerate(zip(confidences, predictions, labels)):
                if label == -100:
                    continue
                    
                is_correct = (pred == label)
                
                all_confidences.append(conf)
                all_correctness.append(is_correct)
                token_idx += 1
        
        # Plot confidence distributions
        correct_conf = [c for c, correct in zip(all_confidences, all_correctness) if correct]
        wrong_conf = [c for c, correct in zip(all_confidences, all_correctness) if not correct]
        
        plt.figure(figsize=(12, 6))
        plt.hist(correct_conf, bins=20, alpha=0.5, label='Correct Predictions')
        plt.hist(wrong_conf, bins=20, alpha=0.5, label='Incorrect Predictions')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.title('Confidence Distribution: Correct vs Incorrect Predictions')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'confidence_analysis.png'))
        plt.show()
        
        # Calculate metrics by confidence threshold
        thresholds = np.arange(0.5, 1.0, 0.05)
        results = []
        
        for threshold in thresholds:
            # Count predictions above threshold
            above_threshold = [correct for conf, correct in zip(all_confidences, all_correctness) 
                            if conf >= threshold]
            
            if above_threshold:
                accuracy = sum(above_threshold) / len(above_threshold)
                coverage = len(above_threshold) / len(all_confidences)
                results.append((threshold, accuracy, coverage))
        
        # Plot threshold analysis
        if results:  # Check if we have results to plot
            thresholds, accuracies, coverages = zip(*results)
            
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            ax1.plot(thresholds, accuracies, 'b-', label='Accuracy')
            ax1.set_xlabel('Confidence Threshold')
            ax1.set_ylabel('Accuracy', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            
            ax2 = ax1.twinx()
            ax2.plot(thresholds, coverages, 'r-', label='Coverage')
            ax2.set_ylabel('Coverage', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            fig.tight_layout()
            plt.title('Accuracy vs Coverage by Confidence Threshold')
            
            # Add legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
            
            plt.savefig(os.path.join(results_dir, 'threshold_analysis.png'))
            plt.show()
        
        return {"mean_confidence_correct": np.mean(correct_conf), 
                "mean_confidence_incorrect": np.mean(wrong_conf) if wrong_conf else 0}