import os
import re
# Project directory configuration
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_DIR, "scripts")
UTILS_DIR = os.path.join(PROJECT_DIR, "utils")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

# Data paths
DATASET_PATH = os.path.join(DATA_DIR, "tokenized_train")
BEST_PARAMS_PATH = os.path.join(PROJECT_DIR, "optimization", "optimization_results.json")
TRAINING_RESULTS_PATH = os.path.join(LOGS_DIR, "training_results.json")

# Model configuration
MODEL_NAME = "roberta-base"
MODEL_OUTPUT_DIR = os.path.join(MODELS_DIR, "roberta-finetuned-ner")
LABELS = {"O": 0, "B-PERSON": 1, "I-PERSON": 2, "TITLE": 3}
#LABELS = {"O": 0, "B-PERSON": 1, "I-PERSON": 2}
NUM_LABELS = len(LABELS)
ID2LABEL = {str(i): label for label, i in LABELS.items()}
PERSON_TAG_PATTERN = re.compile(r'(^|B-|I-)(per(son)?|PER)$', re.IGNORECASE)
TITLES_array = {"Mr.", "Mrs.", "Miss", "Ms.", "Dr.", "Prof.", "Sir", "Madam",
          "President", "Chancellor", "Minister", "Mayor", "King", "Queen", "Pope", "Chief",
          "Lord", "Lady", "Baron", "Duke", "Duchess", "Count", "Countess", "Prince",
          "Princess", "Emperor", "Empress", "Manager", "Director", "CEO", "CFO", "CTO", "COO", "Vice President",
          "Secretary", "Treasurer", "Chairman", "Chairwoman", "Leader", "Commander",
          "General", "Admiral", "Colonel", "Major", "Captain", "Lieutenant", "Sergeant",
          "Corporal", "Private", "Officer", "Agent", "Detective", "Inspector", 
          "Deputy", "Sheriff", "Marshal", "Warden", "Guard", "Watchman", "Bouncer",
          
          # Academic/Professional
          "Dean", "Provost", "Principal", "Regent", "Professor Emeritus", "Associate Professor", 
          "Assistant Professor", "Fellow", "Lecturer", "Instructor", "Tutor",
          
          # Religious
          "Bishop", "Cardinal", "Reverend", "Pastor", "Priest", "Rabbi", "Imam", "Deacon", 
          "Archbishop", "Chaplain", "Father", "Sister", "Brother", "Elder",
          
          # Military/Law Enforcement
          "Brigadier", "Commander", "Ensign", "Cadet", "Constable", "Commissioner", 
          "Investigator", "Staff Sergeant", "Petty Officer", "Warrant Officer",
          
          # Political/Governmental
          "Governor", "Senator", "Congressman", "Congresswoman", "Ambassador", "Councilor", 
          "Councillor", "Representative", "Delegate", "Commissioner", "Attorney General",
          
          # Nobility/Royalty
          "Viscount", "Viscountess", "Earl", "Marquess", "Marchioness", "Baronet", 
          "Knight", "Dame", "Esquire", "Sultan", "Emir", "Tsar", "Czar", "Maharaja", "Maharani",
          
          # Medical/Legal
          "Judge", "Justice", "Magistrate", "Surgeon", "Nurse", "Physician", 
          "Attorney", "Barrister", "Solicitor",
          
          # Corporate
          "Executive", "Supervisor", "Coordinator", "Associate", "Specialist", "Analyst", 
          "Manager", "Superintendent", "Foreman", "Forewoman", "Head", "Lead"
}
TITLES = {title.lower() for title in TITLES_array}
AVOIDED_symbols = {"-", "=", "+", ":", ";", ",", ".", "!", "?", "@", "#", "$", "%", "^", "&", "*"}
# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

