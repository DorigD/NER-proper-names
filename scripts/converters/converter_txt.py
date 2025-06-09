import json
import os
import sys
import re
from utils.config import PERSON_TAG_PATTERN, TITLES

LABELS = {"O": 0, "B-PERSON": 1, "I-PERSON": 2, "TITLE": 3}
def convert_txt_to_json(txt_file_path, json_file_path, replace=True):
    dataset = []
    current_sentence = {"tokens": [], "ner_tags": []}
    line_count = 0
    skipped_lines = 0
    person_tag_pattern = PERSON_TAG_PATTERN

    try:
        with open(txt_file_path, "r", encoding="utf-8") as file:
            for line in file:
                line_count += 1
                line = line.strip()
                
                # Handle new sentences (blank lines)
                if not line:
                    if current_sentence["tokens"]:
                        dataset.append(current_sentence)
                        current_sentence = {"tokens": [], "ner_tags": []}
                    continue

                parts = line.split()
                if not parts:  # Skip empty lines
                    continue
                    
                # Always take the first item as the word/token
                word = parts[0]
                
                # First determine what the original tag would be
                original_tag = "O"  # Default to O
                found_person_tag = False
                
                # First pass: look for person tags
                for part in parts[1:]:
                    if person_tag_pattern.match(part):
                        original_tag = part
                        found_person_tag = True
                        break
                
                # Only look for other entity tags if no person tag was found
                if not found_person_tag:
                    for part in parts[1:]:
                        if part != "O" and re.match(r'(B|I)-\w+', part):
                            original_tag = part
                            break
                
                # Check if word is a title AND tag is either O or PERSON-related
                if word.lower() in TITLES and (original_tag == "O" or person_tag_pattern.match(original_tag)):
                    # Add title with special tag instead of skipping
                    current_sentence["tokens"].append(word)
                    if "TITLE" in LABELS:
                        current_sentence["ner_tags"].append(LABELS["TITLE"])
                    else:
                        current_sentence["ner_tags"].append(LABELS["O"])
                    continue
                
                # Process tag for non-title words
                tag = "O"
                if person_tag_pattern.match(original_tag):
                    if original_tag.startswith("B-"):
                        tag = "B-PERSON"
                    elif original_tag.startswith("I-"):
                        tag = "I-PERSON"
                    else:  # Handle just "PERSON" without B- or I- prefix
                        tag = "B-PERSON"  # Default to beginning
                
            
                                
                # Convert NER tags to integer labels
                ner_label = LABELS.get(tag, LABELS["O"])

                current_sentence["tokens"].append(word)
                current_sentence["ner_tags"].append(ner_label)

        # Save last sentence
        if current_sentence["tokens"]:
            dataset.append(current_sentence)

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
        print(f"Error converting {txt_file_path}: {str(e)}")
        return False




