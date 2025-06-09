import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import sys
import os

# Add the project path to import your custom model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train import RobertaCRFForTokenClassification, SimplifiedRobertaCRFForTokenClassification
from utils.config import MODELS_DIR
from utils.label_config import load_label_config

def load_custom_model(model_path):
    """Load your custom trained model properly with correct label configuration"""
    print(f"Loading custom model from: {model_path}")
    
    # Load label configuration first
    try:
        label_config = load_label_config(model_path)
        ID2LABEL = label_config["id2label"]
        LABEL2ID = label_config["label2id"]
        NUM_LABELS = label_config["num_labels"]
        print(f"Loaded label configuration: {label_config['labels']}")
    except Exception as e:
        print(f"Warning: Could not load label configuration: {e}")
        # Fallback to defaults
        ID2LABEL = {"0": "O", "1": "B-PERSON", "2": "I-PERSON"}
        LABEL2ID = {"O": 0, "B-PERSON": 1, "I-PERSON": 2}
        NUM_LABELS = 3
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Try to load the model in order of preference
    model = None
    model_type = "unknown"
    
    # 1. Try loading as simplified custom model
    try:
        model = SimplifiedRobertaCRFForTokenClassification.from_pretrained_custom(
            model_path, NUM_LABELS, ID2LABEL, LABEL2ID
        )
        print("Loaded SimplifiedRobertaCRFForTokenClassification")
        model_type = "simplified"
    except Exception as e:
        print(f"Failed to load as SimplifiedRobertaCRF: {e}")
    
    # 2. Try loading as complex custom model (this won't work with from_pretrained yet)
    if model is None:
        try:
            # This would require implementing from_pretrained for RobertaCRFForTokenClassification
            print("Complex model loading not yet implemented with from_pretrained")
        except Exception as e:
            print(f"Failed to load as RobertaCRF: {e}")
    
    # 3. Fallback to standard AutoModel
    if model is None:
        try:
            model = AutoModelForTokenClassification.from_pretrained(model_path)
            print("Loaded AutoModelForTokenClassification")
            model_type = "auto"
        except Exception as e:
            print(f"Failed to load as AutoModel: {e}")
            return None, None, None
    
    # Ensure proper config for all model types
    if hasattr(model, 'config'):
        model.config.id2label = ID2LABEL
        model.config.label2id = LABEL2ID
        model.config.num_labels = NUM_LABELS
    
    return model, tokenizer, model_type, ID2LABEL

def simple_test():
    """Simple test with proper model loading"""
    model_path = os.path.join(MODELS_DIR, "roberta-finetuned-ner-v1")
    
    # Load model properly
    result = load_custom_model(model_path)
    if result[0] is None:
        print("Failed to load model")
        return
    
    model, tokenizer, model_type, id2label = result
    
    print(f"Model type: {model_type}")
    print(f"Model config: {getattr(model.config, 'id2label', 'No id2label found')}")
    print(f"Label mapping: {id2label}")
    
    # Test sentences
    test_sentences = [
        "John Smith went to the store.",
        "Mary Johnson and Bob Wilson are friends.",
        "The cat sat on the mat.",
        "Alice Cooper visited New York yesterday."
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    for sentence in test_sentences:
        print(f"\n" + "="*40)
        print(f"Testing: '{sentence}'")
        print("="*40)
        
        # Tokenize
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # Predict
        with torch.no_grad():
            if model_type in ["simplified", "complex"]:
                # Use custom model's forward method
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                if 'predictions' in outputs:
                    predictions = outputs['predictions']
                else:
                    predictions = torch.argmax(outputs['logits'], dim=-1)
            else:
                # Standard model
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Get tokens and predictions
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        pred_ids = predictions[0].cpu().numpy()
        
        # Display results
        print("Token\t\t\tPrediction")
        print("-" * 40)
        
        person_count = 0
        for token, pred_id in zip(tokens, pred_ids):
            clean_token = token.replace('Ġ', ' ').replace('</s>', 'EOS').replace('<s>', 'BOS')
            if clean_token.strip():  # Skip empty tokens
                # Handle both string and int keys
                pred_label = id2label.get(str(pred_id), f"UNKNOWN_{pred_id}")
                print(f"{clean_token:20}\t{pred_label}")
                
                if pred_label in ['B-PERSON', 'I-PERSON']:
                    person_count += 1
        
        print(f"\nPerson tokens detected: {person_count}")

if __name__ == "__main__":
    simple_test()