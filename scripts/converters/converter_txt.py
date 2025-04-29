import json
import os
import sys
import re

# Define valid labels (Only keeping PERSON tags)
LABELS = {"O": 0, "B-PERSON": 1, "I-PERSON": 2, "TITLE": 3}

# List of titles to remove
TITLES = {"Mr.", "Mrs.", "Miss", "Ms.", "Dr.", "Prof.", "Sir", "Madam",
          "President", "Chancellor", "Minister", "Mayor", "King", "Queen", "Pope"}


def convert_txt_to_json(txt_file_path, json_file_path, replace=True):
    dataset = []
    current_sentence = {"tokens": [], "ner_tags": []}
    line_count = 0
    skipped_lines = 0
    person_tag_pattern = re.compile(r'.*PER.*')

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
                
                # Check if word is a title
                if word in TITLES:
                    # Add title with special tag instead of skipping
                    current_sentence["tokens"].append(word)
                    current_sentence["ner_tags"].append(LABELS["TITLE"])
                    continue
                
                # Find a tag containing "PERSON" or default to "O"
                tag = "O"
                for part in parts[1:]:  # Look through all columns after the word
                    if person_tag_pattern.match(part):
                        if part.startswith("B-"):
                            tag = "B-PERSON"
                            break
                        elif part.startswith("I-"):
                            tag = "I-PERSON"
                            break
                        else:  # Handle just "PERSON" without B- or I- prefix
                            tag = "B-PERSON"  # Default to beginning
                            break
                
            
                                
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




