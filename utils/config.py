import os

# Project directory configuration
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_DIR, "scripts")
UTILS_DIR = os.path.join(PROJECT_DIR, "utils")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

# Data paths
DATASET_PATH = os.path.join(DATA_DIR, "tokenized_train")
BEST_PARAMS_PATH = os.path.join(LOGS_DIR, "optuna_study_results.json")
TRAINING_RESULTS_PATH = os.path.join(LOGS_DIR, "training_results.json")

# Model configuration
MODEL_NAME = "roberta-base"
MODEL_OUTPUT_DIR = os.path.join(MODELS_DIR, "roberta-finetuned-ner")
NUM_LABELS = 4  # O, B-PERSON, I-PERSON, TITLE
LABELS = {"O": 0, "B-PERSON": 1, "I-PERSON": 2, "TITLE": 3}
ID2LABEL = {str(i): label for label, i in LABELS.items()}

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

