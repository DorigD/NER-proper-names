import torch
from torch.nn import functional as F
from torch import nn
from transformers.modeling_outputs import TokenClassifierOutput

import os
import json
import torch
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification

def load_ner_model(model_path, use_adapters=False, use_focal_loss=False, use_crf=False):
    """
    Load a NER model with optional adapter, focal loss, or CRF
    
    Args:
        model_path: Path to the saved model directory
        use_adapters: Whether the model uses PEFT adapters
        use_focal_loss: Whether to wrap model with focal loss
        use_crf: Whether to wrap model with CRF layer
        
    Returns:
        Loaded model with appropriate wrappers
    """
    from utils.config import LABELS, NUM_LABELS
    
    # Check if we're loading a PEFT adapter model
    if use_adapters:
        # Load the adapter config
        try:
            with open(os.path.join(model_path, "adapter_config.json")) as f:
                adapter_config = json.load(f)
            
            base_model_name = adapter_config.get("base_model", "roberta-base")
            
            # Load the base model first
            base_model = AutoModelForTokenClassification.from_pretrained(
                base_model_name,
                num_labels=NUM_LABELS,
                id2label={v: k for k, v in LABELS.items()},
                label2id=LABELS,
            )
            
            # Then load the PEFT adapter
            from peft import PeftModel
            model = PeftModel.from_pretrained(base_model, model_path)
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Could not find valid adapter config: {e}. Loading as standard model.")
            # Fallback to standard model loading
            model = AutoModelForTokenClassification.from_pretrained(model_path)
    else:
        # Standard model loading
        model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    # Wrap with CRF if needed
    if use_crf:
        try:
            from torchcrf import CRF
            model = NERModelWithCRF(model, NUM_LABELS)
        except ImportError:
            print("Warning: torchcrf not installed. Skipping CRF layer.")
    
    # Wrap with FocalLoss if needed
    if use_focal_loss and not use_crf:
        # Load model config to get focal parameters
        try:
            with open(os.path.join(model_path, "model_config.json")) as f:
                model_config = json.load(f)
            
            alpha = model_config.get("focal_alpha", 1.0)
            gamma = model_config.get("focal_gamma", 2.0)
            model = NERModelWithFocalLoss(model, alpha=alpha, gamma=gamma)
        except (FileNotFoundError, json.JSONDecodeError):
            # Use default parameters
            model = NERModelWithFocalLoss(model)
    
    model.eval() 
    return model

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                labels.view(-1),
                                reduction='none',
                                ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class NERModelWithFocalLoss(torch.nn.Module):
    def __init__(self, base_model, alpha=1.0, gamma=2.0, person_weight=5.0):
        super().__init__()
        self.model = base_model
        self.config = base_model.config if hasattr(base_model, "config") else None
        self.gamma = gamma
        
        # Create class weights tensor with higher weight for PERSON tags
        num_labels = self.config.num_labels if self.config else base_model.num_labels
        self.alpha = torch.ones(num_labels)
        
        # Find PERSON tag ID from id2label mapping
        if hasattr(self.config, "id2label"):
            for idx, label in self.config.id2label.items():
                if "PERSON" in label:
                    self.alpha[int(idx)] = person_weight
        
        # Move to same device as model
        device = next(base_model.parameters()).device
        self.alpha = self.alpha.to(device)
    
    def forward(self, **inputs):
        # Filter out inputs the base model doesn't expect
        allowed_inputs = [
            "input_ids", "attention_mask", "token_type_ids", 
            "position_ids", "inputs_embeds", "labels", "output_attentions", 
            "output_hidden_states", "return_dict"
        ]
        model_inputs = {k: v for k, v in inputs.items() if k in allowed_inputs}
        
        # Forward through base model
        outputs = self.model(**model_inputs)
        logits = outputs.logits
        loss = None
        
        if "labels" in inputs:
            labels = inputs["labels"]
            # Standard cross entropy (no reduction)
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction='none',
                ignore_index=-100
            )
            
            # Get probabilities
            pt = torch.exp(-ce_loss)
            
            # Calculate focal weights
            focal_weights = (1 - pt) ** self.gamma
            
            # Get class weights for each token
            label_indices = labels.view(-1)
            valid_indices = label_indices != -100
            
            # Apply alpha weights based on class
            alpha_weights = torch.ones_like(focal_weights)
            if valid_indices.any():
                valid_labels = label_indices[valid_indices]
                alpha_weights[valid_indices] = self.alpha[valid_labels]
            
            # Combine for final loss
            focal_loss = alpha_weights * focal_weights * ce_loss
            loss = focal_loss.mean()
        
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )

class NERModelWithCRF(torch.nn.Module):
    def __init__(self, base_model, num_labels):
        super().__init__()
        self.model = base_model
        self.num_labels = num_labels
        self.config = base_model.config if hasattr(base_model, "config") else None
        
        # Import CRF inside method to avoid errors if not installed
        from torchcrf import CRF
        self.crf = CRF(num_labels, batch_first=True)
        
    def forward(self, **inputs):
        # Filter out inputs the base model doesn't expect
        allowed_inputs = [
            "input_ids", "attention_mask", "token_type_ids", 
            "position_ids", "inputs_embeds", "labels", "output_attentions", 
            "output_hidden_states", "return_dict"
        ]
        model_inputs = {k: v for k, v in inputs.items() if k in allowed_inputs}
        
        # Get device of inputs
        device = next(self.model.parameters()).device
        
        # Move CRF to device if needed
        if next(self.crf.parameters()).device != device:
            self.crf = self.crf.to(device)
        
        # Get logits from base model
        outputs = self.model(**model_inputs)
        logits = outputs.logits
        
        # For training with CRF
        if "labels" in inputs:
            # Create mask from attention mask, we only want to consider real tokens
            mask = inputs["attention_mask"].bool() if "attention_mask" in inputs else None
            
            # Get labels and ensure they are valid for CRF (no -100 values)
            labels = inputs["labels"].clone()
            
            # Ensure everything is on the same device
            if labels.device != device:
                labels = labels.to(device)
            if mask is not None and mask.device != device:
                mask = mask.to(device)
            if logits.device != device:
                logits = logits.to(device)
            
            # Create a proper mask that excludes -100 labels
            if mask is not None:
                # Set positions with -100 to False in mask 
                mask = mask & (labels >= 0)
            
            # Replace -100 labels with 0 (these will be masked anyway)
            labels[labels == -100] = 0
            
            # Calculate CRF loss with proper masking
            try:
                # Compute loss (negative log-likelihood)
                loss = -self.crf(logits, labels, mask=mask, reduction='mean')
            except Exception as e:
                # Fall back to cross-entropy loss
                from torch.nn import CrossEntropyLoss
                loss_fct = CrossEntropyLoss()
                active_loss = mask.view(-1) if mask is not None else None
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, 
                    labels.view(-1), 
                    torch.tensor(loss_fct.ignore_index, device=device).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
                print("Falling back to CrossEntropyLoss")
            
            return TokenClassifierOutput(loss=loss, logits=logits)
        else:
            # For inference: decode most likely sequence
            mask = inputs["attention_mask"].bool() if "attention_mask" in inputs else None
            
            # Ensure everything is on the same device
            if mask is not None and mask.device != device:
                mask = mask.to(device)
            if logits.device != device:
                logits = logits.to(device)
            
            # Get best path from CRF
            best_path = self.crf.decode(logits, mask=mask)
            
            # Convert list of lists to tensor
            batch_size, seq_len, _ = logits.shape
            
            # Create tensor directly on right device
            decoded = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
            
            # Process list of paths safely
            for b, path in enumerate(best_path):
                # Skip empty paths
                if not path:
                    continue
                    
                # Create tensor on correct device
                path_len = len(path)
                path_tensor = torch.tensor(path, dtype=torch.long, device=device)
                
                # Ensure we don't exceed sequence length
                valid_len = min(path_len, seq_len)
                decoded[b, :valid_len] = path_tensor[:valid_len]
            
            # Return with everything on the same device
            return TokenClassifierOutput(logits=logits, predictions=decoded)

class SafeDataCollatorForTokenClassification(DataCollatorForTokenClassification):
    def __call__(self, features):
        try:
            # Standard collation
            return super().__call__(features)
        except Exception as e:
            # Fallback for inconsistent structures
            import torch
            
            # Force consistent dimensions
            max_length = max(len(x["input_ids"]) for x in features)
            
            batch = {
                "input_ids": [],
                "attention_mask": [],
                "labels": []
            }
            
            for feature in features:
                # Pad each feature to max_length
                input_ids = feature["input_ids"]
                attn_mask = feature["attention_mask"]
                labels = feature["labels"]
                
                padding_length = max_length - len(input_ids)
                
                # Add padding
                input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                attn_mask = attn_mask + [0] * padding_length
                labels = labels + [-100] * padding_length
                
                batch["input_ids"].append(input_ids)
                batch["attention_mask"].append(attn_mask)
                batch["labels"].append(labels)
            
            # Convert to tensors
            batch = {k: torch.tensor(v) for k, v in batch.items()}
            return batch