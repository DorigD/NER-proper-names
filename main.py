# First, import torch completely before anything else
import torch
import numpy as np
import os
import json

# Then import transformers
from transformers import AutoModelForTokenClassification, AutoTokenizer
import transformers
# Finally import your local modules
from utils.config import PROJECT_DIR
from scripts.train import train_model
from scripts.preprocess import preprocess
from scripts.converters.Converter import convert_file
transformers.logging.set_verbosity_error() 
import warnings
warnings.filterwarnings("ignore")
from utils.models import AdapterWithCRF
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
        Load the model and tokenizer with proper handling of pre-trained NER models.
        """
        if model_path is None:
            model_path = os.path.join(PROJECT_DIR, "models", "roberta-finetuned-ner")
        
        # Check if model_path is a local path or a model ID
        is_local_path = os.path.exists(model_path)
        is_pretrained_ner = "ner" in model_path.lower() if not is_local_path else False
        
        # Load default labels from config
        from utils.config import ID2LABEL
        default_id2label = ID2LABEL
        
        # Get the model's configuration first for pre-trained NER models
        if is_pretrained_ner:
            from transformers import AutoConfig
            print(f"Loading pre-trained NER model config: {model_path}")
            try:
                config = AutoConfig.from_pretrained(model_path)
                if hasattr(config, "id2label"):
                    # Use the pre-trained model's labels directly
                    self.id2label = {str(k): v for k, v in config.id2label.items()}
                    print(f"Using pre-trained model's label mapping: {self.id2label}")
                else:
                    # Unlikely, but fallback to default
                    self.id2label = default_id2label
                    print("Pre-trained model has no id2label mapping, using default")
            except Exception as e:
                print(f"Error loading pre-trained model config: {e}")
                self.id2label = default_id2label
                
            # Now load the model with its original configuration
            print(f"Loading pre-trained NER model with its original configuration")
            self.model = AutoModelForTokenClassification.from_pretrained(model_path).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Update id2label from actually loaded model to be safe
            self.id2label = {str(k): v for k, v in self.model.config.id2label.items()}
            print(f"Final id2label mapping: {self.id2label}")
            
        else:
            # The rest of your existing code for non-NER models
            if is_local_path:
                # Load from local path with model_config.json
                try:
                    with open(os.path.join(model_path, "model_config.json"), "r") as f:
                        model_config = json.load(f)
                        self.id2label = model_config["id2label"]
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Warning: Could not load model_config.json: {e}")
                    # Use default labels
                    self.id2label = default_id2label
            else:
                # For Hugging Face models, load and extract the id2label from the config
                print(f"Loading model from Hugging Face Hub: {model_path}")
                from transformers import AutoConfig
                try:
                    config = AutoConfig.from_pretrained(model_path)
                    if hasattr(config, "id2label"):
                        # Check if this is a compatible NER model
                        ner_labels = set(['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC', 
                                         'B-PERSON', 'I-PERSON'])
                        config_labels = set(config.id2label.values())
                        
                        if len(ner_labels.intersection(config_labels)) >= 3 or len(config.id2label) == len(default_id2label):
                            self.id2label = config.id2label
                            print(f"Using model's label mapping: {self.id2label}")
                        else:
                            print(f"Model has different id2label mapping: {config.id2label}")
                            print("Using default NER labels instead")
                            self.id2label = default_id2label
                    else:
                        print("Model has no id2label mapping, using default NER labels")
                        self.id2label = default_id2label
                except Exception as e:
                    print(f"Error loading model config from Hub: {e}")
                    print("Using default NER labels instead")
                    self.id2label = default_id2label
            
            # Ensure our ID2LABEL keys match the loaded model's expected range
            normalized_id2label = {str(i): label for i, label in enumerate(self.id2label.values())}
            self.id2label = normalized_id2label
            
            # Load model with our configuration
            if not is_local_path:
                # When loading from HuggingFace, force our label mapping
                print(f"Enforcing custom NER label mapping for {model_path}")
                config = AutoConfig.from_pretrained(
                    model_path,
                    num_labels=len(self.id2label),
                    id2label=self.id2label,
                    label2id={label: int(id) for id, label in self.id2label.items()}
                )
                self.model = AutoModelForTokenClassification.from_pretrained(
                    model_path, 
                    config=config
                ).to(self.device)
            else:
                # For local models, load directly
                self.model = AutoModelForTokenClassification.from_pretrained(model_path).to(self.device)
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.model.eval()    
        return self

    def train(self, training_data=None, replace=True, optimize=True, save_train="default"):
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
                        convert_file(os.path.join(training_data, file), replace=replace)
                        break
            else:
                convert_file(training_data, replace=replace)
        except Exception as e:
            print(f"Error during data conversion: {e}")
            return
        try:
            # Preprocess the training data
            preprocess(ds_name=save_train)
        except Exception as e:
            print(f"Error during data preprocessing: {e}")
            return
        if optimize:
            # Optimize hyperparameters using Optuna
            try:
                print("---------------Optimizing hyperparameters using Optuna--------------")
                self.optimize_hyperparameters()
            except Exception as e:
                print(f"Error during hyperparameter optimization: {e}")
                return
            
        # Always pass a model name string, not a model object
        model_name = self.base_model
        # If model exists, extract the model name from its config
        if self.model and hasattr(self.model.config, "_name_or_path"):
            model_name = self.model.config._name_or_path
        
        return train_model(model_name)

    def set_model(self, model_path):
        """
        Set the model for the NER class.

        :param model_path: Path to the pre-trained model.
        """
        self.base_model = model_path
    
    def optimize_hyperparameters(self, n_trials=16, visualize=False):
        """
        Optimize hyperparameters for model training using Optuna.
        
        Args:
            n_trials: Number of optimization trials to run (default: 16)
            visualize: Whether to display optimization visualizations (default: False)
            
        Returns:
            Dictionary with best hyperparameters and optimization results
        """
        import optuna
        from transformers import (
            AutoModelForTokenClassification, 
            Trainer, 
            TrainingArguments,
            DataCollatorForTokenClassification,
            EvalPrediction,
            EarlyStoppingCallback
        )
        from datasets import load_from_disk
        from sklearn.metrics import precision_recall_fscore_support
        import numpy as np
        import os
        import json
        import tempfile
        from datetime import datetime
        from utils.config import PROJECT_DIR, DATASET_PATH, BEST_PARAMS_PATH
        from utils.metrics import compute_metrics
        from adapters import AutoAdapterModel, AdapterConfig
        from transformers import AdapterTrainer
        # Ensure we have a tokenized dataset
        dataset_path = DATASET_PATH
        if not os.path.exists(dataset_path):
            from scripts.preprocess import preprocess
            preprocess()
            
        # Load tokenized dataset
        print("Loading tokenized dataset...")
        tokenized_dataset = load_from_disk(dataset_path)
        
        # Create only one results directory for final results
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_dir = os.path.join(PROJECT_DIR, "optimization")
        os.makedirs(results_dir, exist_ok=True)
        
        # Load tokenizer if not already loaded
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            
        # Define the optimization objective
        def objective(trial):
            adapter_name = "ner_adapter"
            num_train_epochs = trial.suggest_int("num_train_epochs", 3, 10)
            per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32])
            weight_decay = trial.suggest_float("weight_decay", 0.0, 0.3)
            learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
            
            # Add these parameters to trial
            person_weight = trial.suggest_float("person_weight", 1.0, 7.0)  
            gamma = trial.suggest_float("gamma", 0.5, 3.0)
            i_person_ratio = trial.suggest_float("i_person_ratio", 0.4, 0.8)
            warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
            gradient_accumulation_steps = trial.suggest_int("gradient_accumulation_steps", 1, 4)
            
            # Add adapter-specific hyperparameters
            adapter_size = trial.suggest_categorical("adapter_size", [16, 32, 64])
            reduction_factor = trial.suggest_int("reduction_factor", 8, 32)
            adapter_type = trial.suggest_categorical("adapter_type", ["pfeiffer", "houlsby"])
            
            # Add to objective function in main.py
            title_weight = trial.suggest_float("title_weight", 0.8, 2.5)

            # Add this with your other trial.suggest parameters
            person_scale = trial.suggest_float("person_scale", 1.0, 3.0)

            # Add CRF-specific hyperparameters
            use_crf = trial.suggest_categorical("use_crf", [True, False])
            crf_learning_rate = trial.suggest_float("crf_learning_rate", 5e-6, 2e-4, log=True) if use_crf else None

            # Use a temporary directory for this trial that will be cleaned up automatically
            with tempfile.TemporaryDirectory() as trial_dir:
                print(f"Trial {trial.number} using temporary directory: {trial_dir}")
                
                # Get the number of labels from the config
                from utils.config import ID2LABEL, LABELS
                num_labels = len(ID2LABEL)
                
                if use_crf:
                    print(f"Trial {trial.number}: Using CRF model")
                    # Create CRF model for this trial
                    model = AdapterWithCRF(
                        model_name=self.base_model,
                        num_labels=num_labels,
                        adapter_name=adapter_name,
                        adapter_size=adapter_size,
                        reduction_factor=reduction_factor
                    )
                else:
                    print(f"Trial {trial.number}: Using standard adapter model")
                    # Prepare model with adapters (existing code)
                    model = AutoAdapterModel.from_pretrained(self.base_model)
                    # Add task adapter for NER
                    adapter_config = AdapterConfig(
                        mh_adapter=True,
                        output_adapter=True,
                        reduction_factor=reduction_factor,
                        non_linearity="relu",
                        adapter_size=adapter_size
                    )
                    model.add_adapter(adapter_name, config=adapter_config)
                    # Add classification head
                    model.add_classification_head(
                        adapter_name,
                        num_labels=num_labels,
                        id2label={str(i): label for i, label in enumerate(LABELS)}
                    )
                    # Activate adapter for training
                    model.train_adapter(adapter_name)
                
                # Apply class weights during training
                class_weights = torch.ones(num_labels)
                class_weights[0] = 0.2  # Reduce weight for "O" tag
                class_weights[1] = person_weight  # Use the optimized person weight
                class_weights[2] = person_weight * i_person_ratio  # I-PERSON slightly lower
                if "TITLE" in LABELS:
                    title_idx = list(LABELS.keys()).index("TITLE")
                    class_weights[title_idx] = title_weight
                
                model.config.class_weights = class_weights.tolist()
                
                # Data collator for NER
                data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
                
                # Define training arguments with minimal disk usage
                if use_crf:
                    training_args = TrainingArguments(
                        output_dir=trial_dir,
                        num_train_epochs=num_train_epochs,
                        per_device_train_batch_size=per_device_train_batch_size,
                        weight_decay=weight_decay,
                        learning_rate=crf_learning_rate,
                        eval_strategy="epoch",
                        save_strategy="no", 
                        logging_strategy="no",
                        load_best_model_at_end=True, 
                        metric_for_best_model="b_person_f1",
                        greater_is_better=True,
                        max_grad_norm=1.0,
                        per_device_eval_batch_size=8,
                        report_to="none",
                    )
                else:
                    training_args = TrainingArguments(
                        output_dir=trial_dir,
                        num_train_epochs=num_train_epochs,
                        per_device_train_batch_size=per_device_train_batch_size,
                        weight_decay=weight_decay,
                        learning_rate=learning_rate,
                        eval_strategy="epoch",
                        save_strategy="no", 
                        logging_strategy="no",
                        load_best_model_at_end=True, 
                        metric_for_best_model="b_person_f1",
                        greater_is_better=True,
                        max_grad_norm=1.0,
                        per_device_eval_batch_size=8,
                        report_to="none",
                    )
                
                # Create a custom trainer with focal loss
                class FocalLossTrainer(AdapterTrainer):  # Use AdapterTrainer, not Trainer
                    def __init__(self, gamma, person_scale, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        self.gamma = gamma
                        self.person_scale = person_scale

                    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                        labels = inputs.pop("labels")
                        outputs = model(**inputs, adapter_names=[adapter_name])
                        logits = outputs.logits
                        
                        # Get the mask for valid positions (ignore padding tokens)
                        active_loss = labels.view(-1) != -100
                        active_logits = logits.view(-1, num_labels)
                        active_labels = torch.where(
                            active_loss, 
                            labels.view(-1), 
                            torch.tensor(0).type_as(labels)
                        )
                        
                        # Apply softmax to get probabilities
                        probs = torch.nn.functional.softmax(active_logits, dim=-1)
                        
                        # Get probabilities for the true class
                        pt = probs.gather(1, active_labels.unsqueeze(1)).squeeze()
                        
                        # Apply class weights
                        weights = class_weights.to(model.device)
                        alpha = weights.gather(0, active_labels)
                        
                        # Calculate focal loss: -α * (1-pt)^γ * log(pt)
                        focal_weight = (1 - pt).pow(self.gamma)
                        focal_loss = -alpha * focal_weight * torch.log(pt + 1e-10)
                        
                        # FIRST apply person scale before calculating final loss!
                        person_indices = (active_labels == LABELS["B-PERSON"]) | (active_labels == LABELS["I-PERSON"])
                        if person_indices.any():
                            focal_loss[person_indices] *= self.person_scale
                        
                        # THEN apply mask and compute mean
                        loss = torch.zeros_like(labels.view(-1), dtype=torch.float)
                        loss[active_loss] = focal_loss[active_loss]
                        loss = loss.sum() / active_loss.sum()
                        
                        return (loss, outputs) if return_outputs else loss
                
                # After creating the model:

                # Create appropriate trainer based on model type
                if use_crf:
                    class CRFTrainer(AdapterTrainer):
                        def __init__(self, gamma=2.0, person_scale=1.5, *args, **kwargs):
                            self.gamma = gamma
                            self.person_scale = person_scale
                            super().__init__(*args, **kwargs)
                        
                        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                            labels = inputs.pop("labels")
                            
                            # For CRF model
                            outputs = model(
                                input_ids=inputs["input_ids"],
                                attention_mask=inputs["attention_mask"],
                                labels=labels
                            )
                            loss = outputs["loss"]
                            
                            # Apply class-specific weighting if needed
                            if self.training:
                                # Apply weighting to CRF loss - simplified as we can't easily get token-wise losses
                                person_ratio = ((labels == LABELS["B-PERSON"]) | (labels == LABELS["I-PERSON"])).sum() / (labels != -100).sum()
                                if person_ratio > 0:
                                    loss = loss * (1 + (self.person_scale - 1) * person_ratio)
                            
                            return (loss, outputs) if return_outputs else loss
                    
                    trainer = CRFTrainer(
                        gamma=gamma,
                        person_scale=person_scale,
                        model=model,
                        args=training_args,
                        train_dataset=tokenized_dataset["train"],
                        eval_dataset=tokenized_dataset["validation"],
                        tokenizer=self.tokenizer,
                        data_collator=data_collator,
                        compute_metrics=compute_metrics
                    )
                else:
                    # Use existing FocalLossTrainer for non-CRF models
                    trainer = FocalLossTrainer(
                        gamma=gamma,
                        person_scale=person_scale,
                        model=model,
                        args=training_args,
                        train_dataset=tokenized_dataset["train"],
                        eval_dataset=tokenized_dataset["validation"],
                        tokenizer=self.tokenizer,
                        data_collator=data_collator,
                        compute_metrics=compute_metrics
                    )
                
                trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))
                
                # Train the model
                trainer.train()
                
                # Evaluate the model
                eval_results = trainer.evaluate()
                
                # Clean up GPU memory after each trial
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                return eval_results["b_person_f1"]
        
        # Check for existing study to continue
        storage_name = f"sqlite:///{os.path.join(results_dir, 'optuna.db')}"
        # Use timestamp in study name to make each run unique
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        study_name = f"ner_optimization_{timestamp}"

        # Just keep the first study creation
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=False,  # Always create new study
            direction="maximize",
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=1, max_resource=10, reduction_factor=3
            )
        )
        
        try:
            # Run optimization
            study.optimize(objective, n_trials=n_trials)
            
            # Get best parameters
            best_params = study.best_params
            best_value = study.best_value
            
            # Save only the final optimization results
            results = {
                "best_params": best_params,
                "best_f1_score": best_value,
                "base_model": self.base_model,
                "timestamp": timestamp,
                "n_trials": n_trials,
            }
            
            os.makedirs(os.path.dirname(BEST_PARAMS_PATH), exist_ok=True)
            with open(BEST_PARAMS_PATH, "w") as f:
                json.dump(results, f, indent=2)
            
            # Add after optimization completes
            if visualize:
                try:
                    from optuna.visualization import plot_param_importances
                    fig = plot_param_importances(study)
                    fig.write_image(os.path.join(results_dir, "param_importances.png"))
                except ImportError:
                    print("Visualization requires plotly. Install with: pip install plotly")
            
            # Visualization if requested
            if visualize:
                # Print results only
                print(f"\nOptimization Results:\n{'-'*20}")
                print(f"Best F1 Score: {best_value:.4f}")
                print("Best hyperparameters:")
                for param, value in best_params.items():
                    print(f"  {param}: {value}")
            
            print(f"Optimization complete. Results saved to {results_dir}")
            return results
        except Exception as e:
            print(f"Optimization failed with error: {e}")
            return {"error": str(e)}
    
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
            
            # Apply advanced post-processing to fix common name boundary issues
           
            
            batch_results.append(entities)
        
        return batch_results

    def evaluate_transformers(self, test_dataset=None, visualize=True):
        """
        Evaluate model performance using Hugging Face's evaluation tools for better compatibility
        with different transformer models, including those not fine-tuned by this code.
        
        Args:
            test_dataset: HuggingFace dataset for testing or path to dataset. If None, will load from default path.
            visualize: Whether to generate and display visualizations
        
        Returns:
            Dictionary of evaluation metrics
        """
        from datasets import load_from_disk
        import evaluate
        from transformers import Trainer, TrainingArguments, DataCollatorForTokenClassification
        import matplotlib.pyplot as plt
        import os
        from datetime import datetime
        import numpy as np
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        from utils.evaluation_utils import convert_numpy_to_python_types
        from utils.config import PROJECT_DIR, DATASET_PATH
        import re
        
        # Load model if not already loaded
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        # Get model name and clean it for use as directory name
        model_name = self.model.config._name_or_path
        clean_model_name = re.sub(r'[\/\\:*?"<>|]', '-', model_name)  # Replace invalid chars
        
        # Create main results directory
        main_results_dir = os.path.join(PROJECT_DIR, "evaluation_results")
        os.makedirs(main_results_dir, exist_ok=True)
        
        # Create model-specific results directory
        results_dir = os.path.join(main_results_dir, clean_model_name)
        os.makedirs(results_dir, exist_ok=True)
        
        # Handle test_dataset - load from disk if a path is provided
        if isinstance(test_dataset, str) and os.path.isdir(test_dataset):
            print(f"Loading dataset from directory: {test_dataset}")
            loaded_dataset = load_from_disk(test_dataset)
            if "test" in loaded_dataset:
                test_dataset = loaded_dataset["test"]
                print(f"Successfully loaded test split with {len(test_dataset)} examples")
            else:
                # Try to get the first split if 'test' is not available
                first_split = next(iter(loaded_dataset.keys()), None)
                if first_split:
                    test_dataset = loaded_dataset[first_split]
                    print(f"No 'test' split found. Using '{first_split}' split with {len(test_dataset)} examples")
                else:
                    print("Failed to find any valid splits in the dataset")
                    return None
        elif test_dataset is None:
            # Load default test dataset
            test_dataset = load_from_disk(DATASET_PATH)["test"]
            print(f"Using default test dataset with {len(test_dataset)} examples")
        
        # Initialize metrics
        seqeval = evaluate.load("seqeval")
        
        # Create a specialized data collator that handles NER data properly
        data_collator = DataCollatorForTokenClassification(
            self.tokenizer, 
            pad_to_multiple_of=8 if torch.cuda.is_available() else None
        )
        
        # Setup trainer for evaluation
        training_args = TrainingArguments(
            output_dir=os.path.join(results_dir, "tmp"),
            per_device_eval_batch_size=8,
            no_cuda=not torch.cuda.is_available(),
        )
        
        # Create a function to compute metrics
        label_list = list(self.id2label.values())
        
        def compute_metrics(eval_preds):
            logits, labels = eval_preds
            predictions = np.argmax(logits, axis=-1)
            
            # Extract valid predictions (ignoring -100 padding tokens)
            true_predictions = []
            true_labels = []
            
            for prediction, label in zip(predictions, labels):
                true_pred = []
                true_label = []
                for p, l in zip(prediction, label):
                    if l != -100:  # Ignore padding tokens
                        true_pred.append(label_list[p])
                        true_label.append(label_list[l])
                true_predictions.append(true_pred)
                true_labels.append(true_label)
            
            # Calculate metrics
            results = seqeval.compute(predictions=true_predictions, references=true_labels)
            
            # Flatten predictions and labels for confusion matrix
            all_preds = [p for preds in true_predictions for p in preds]
            all_labels = [l for labels in true_labels for l in labels]
            
            # Store for later use in visualization
            if not hasattr(compute_metrics, "all_preds"):
                compute_metrics.all_preds = all_preds
                compute_metrics.all_labels = all_labels
            
            return results
        
        # Create Trainer for evaluation with custom data collator
        trainer = Trainer(
            model=self.model,
            args=training_args,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        # Run evaluation
        try:
            eval_results = trainer.evaluate(eval_dataset=test_dataset)
            
            if visualize:
                print(f"Model: {model_name}")
                print("\nOverall Metrics:")
                for metric, value in eval_results.items():
                    if isinstance(value, (int, float)):
                        print(f"{metric}: {value:.4f}")
                
                # Plot confusion matrix
                if hasattr(compute_metrics, "all_preds") and hasattr(compute_metrics, "all_labels"):
                    # Create label mappings for confusion matrix
                    unique_labels = sorted(set(label_list))
                    label_to_id = {label: i for i, label in enumerate(unique_labels)}
                    
                    # Convert string labels to indices
                    y_true = [label_to_id[l] for l in compute_metrics.all_labels]
                    y_pred = [label_to_id[p] for p in compute_metrics.all_preds]
                    
                    # Generate confusion matrix
                    cm = confusion_matrix(y_true, y_pred, labels=range(len(unique_labels)))
                    
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(
                        cm, 
                        annot=True, 
                        fmt='d', 
                        cmap='Blues',
                        xticklabels=unique_labels,
                        yticklabels=unique_labels
                    )
                    plt.xlabel('Predicted')
                    plt.ylabel('True')
                    plt.title(f'Confusion Matrix - {model_name}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
                    plt.show()
            
            # Save evaluation results
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            results = {
                "model_name": model_name,
                "timestamp": timestamp,
                "metrics": eval_results
            }
            
            # Convert for JSON serialization
            serializable_results = convert_numpy_to_python_types(results)
            
            with open(os.path.join(results_dir, f"evaluation_{timestamp}.json"), "w") as f:
                import json
                json.dump(serializable_results, f, indent=2)
            
            if visualize:
                print(f"Evaluation results saved to {results_dir}")
            
            return results
        
        except Exception as e:
            print(f"Error during evaluation with Trainer: {e}")
            print("Falling back to manual evaluation method...")
            return

    # Add this function to better understand your model's errors
    def analyze_predictions(trainer, dataset, tokenizer, id2label):
        print("Analyzing prediction errors...")
        predictions = trainer.predict(dataset)
        pred_labels = np.argmax(predictions.predictions, axis=-1)
        true_labels = predictions.label_ids
        
        error_types = {"false_negative": 0, "false_positive": 0, "boundary_error": 0}
        error_examples = []
        
        for i in range(len(dataset)):
            tokens = dataset[i]["tokens"]
            pred = pred_labels[i]
            true = true_labels[i]
            
            for j in range(len(true)):
                if true[j] == -100:
                    continue
                    
                if true[j] in [1, 2] and pred[j] == 0:  # Missing person
                    error_types["false_negative"] += 1
                    error_examples.append((tokens, j, id2label[str(true[j])], id2label[str(pred[j])]))
                elif true[j] == 0 and pred[j] in [1, 2]:  # False person
                    error_types["false_positive"] += 1
                elif (true[j] == 1 and pred[j] == 2) or (true[j] == 2 and pred[j] == 1):
                    error_types["boundary_error"] += 1
        
        print(f"Error analysis: {error_types}")
        print("\nExample errors:")
        for i, (tokens, pos, true_label, pred_label) in enumerate(error_examples[:5]):
            context = " ".join(tokens[max(0, pos-3):min(len(tokens), pos+4)])
            print(f"Error {i+1}: '{context}' - True: {true_label}, Pred: {pred_label}")
