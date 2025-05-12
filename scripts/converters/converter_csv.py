import json
import os
import sys
import csv
import ast
import re
from utils.config import LABELS, PERSON_TAG_PATTERN, TITLES, SCRIPTS_DIR, PROJECT_DIR


json_dir = os.path.join(PROJECT_DIR, "data", "json")
def convert_csv_to_json(csv_file_path, json_file_path=os.path.join(json_dir,"result.json"), replace=True):
    dataset = []
    sentence_count = 0
    skipped_rows = 0
    # Use case-insensitive regex to match person tags
    person_tag_pattern = PERSON_TAG_PATTERN
   
    try:
        with open(csv_file_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            header = next(reader)  # Skip header
            
            for row in reader:
                try:
                    if len(row) < 4:
                        skipped_rows += 1
                        continue
                    
                    # Extract data from CSV row
                    sentence_num = row[0]
                    sentence_text = row[1]
                    pos_tags = ast.literal_eval(row[2])  # Convert string representation to list
                    ner_tags_original = ast.literal_eval(row[3])  # Convert string representation to list
                    
                    # Create tokens from the sentence
                    tokens = sentence_text.split()
                    
                    current_sentence = {"tokens": [], "ner_tags": []}
                    
                    for i, (token, original_tag) in enumerate(zip(tokens, ner_tags_original)):
                        # Check if token is a title and original tag is either O or PERSON-related
                        if token.lower() in TITLES and (original_tag == "O" or person_tag_pattern.match(original_tag)):
                            # Add title with special tag instead of skipping
                            current_sentence["tokens"].append(token)
                            if "TITLE" in LABELS:
                                current_sentence["ner_tags"].append(LABELS["TITLE"])
                            else:
                                current_sentence["ner_tags"].append(LABELS["O"])
                            continue
                        
                        # Default tag is "O"
                        tag = "O"
                        
                        # Check if the tag contains "per" using case-insensitive regex
                        if person_tag_pattern.match(original_tag):
                            if original_tag.startswith("B-"):
                                tag = "B-PERSON"
                            elif original_tag.startswith("I-"):
                                tag = "I-PERSON"
                            else:  # Handle just "per" without B- or I- prefix
                                tag = "B-PERSON"  # Default to beginning
                                
                                # Check if this continues a previous PERSON entity
                                if i > 0 and person_tag_pattern.match(ner_tags_original[i-1]):
                                    tag = "I-PERSON"
                        
                        # Debug print to verify tokens and tags
                      
                        
                        # Convert NER tags to integer labels
                        ner_label = LABELS.get(tag, LABELS["O"])
                        
                        current_sentence["tokens"].append(token)
                        current_sentence["ner_tags"].append(ner_label)
                    
                    if current_sentence["tokens"]:
                        dataset.append(current_sentence)
                        sentence_count += 1
                
                except Exception as e:
                    print(f"Error processing row: {e}")
                    skipped_rows += 1
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
        
        # Check if file exists and we want to append to it
        if not replace and os.path.exists(json_file_path):
            try:
                with open(json_file_path, "r", encoding="utf-8") as existing_file:
                    existing_data = json.load(existing_file)
                    # Combine existing data with new data
                    dataset = existing_data + dataset
            except json.JSONDecodeError:
                print(f"Warning: Existing file '{json_file_path}' is not valid JSON. Will overwrite.")
        
        # Save to JSON
        with open(json_file_path, "w", encoding="utf-8") as out_file:
            json.dump(dataset, out_file, indent=4, ensure_ascii=False)
        
        action = "Appended to" if not replace and os.path.exists(json_file_path) else "Converted"
        return True
        
    except Exception as e:
        print(f"Error converting {csv_file_path}: {str(e)}")
        return False

