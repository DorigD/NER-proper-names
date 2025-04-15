import json
import os
import sys
import csv
import ast
import re

# Define valid labels (Only keeping PERSON tags)
LABELS = {"O": 0, "B-PERSON": 1, "I-PERSON": 2}

# List of titles to remove
TITLES = {"Mr.", "Mrs.", "Miss", "Ms.", "Dr.", "Prof.", "Sir", "Madam",
          "President", "Chancellor", "Minister", "Mayor", "King", "Queen", "Pope"}

def convert_csv_to_json(csv_file_path, json_file_path):
    dataset = []
    sentence_count = 0
    skipped_rows = 0
    # Use case-insensitive regex to match person tags
    person_tag_pattern = re.compile(r'.*per.*', re.IGNORECASE)
    
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
                    
                    # Ensure token count matches tag count
                    if len(tokens) != len(ner_tags_original):
                        print(f"Warning: Token count ({len(tokens)}) doesn't match tag count ({len(ner_tags_original)}) in sentence {sentence_num}")
                        skipped_rows += 1
                        continue
                    
                    current_sentence = {"tokens": [], "ner_tags": []}
                    
                    for i, (token, original_tag) in enumerate(zip(tokens, ner_tags_original)):
                        # Skip titles
                        if token in TITLES:
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
                        print(f"Word: {token}, Original Tag: {original_tag}, Converted Tag: {tag}")
                        
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
        
        # Save to JSON
        with open(json_file_path, "w", encoding="utf-8") as out_file:
            json.dump(dataset, out_file, indent=4, ensure_ascii=False)
            
        print(f"Converted '{csv_file_path}' to '{json_file_path}'")
        print(f"  - Sentences processed: {sentence_count}")
        print(f"  - Rows skipped: {skipped_rows}")
        return True
        
    except Exception as e:
        print(f"Error converting {csv_file_path}: {str(e)}")
        return False

def main():
    if len(sys.argv) > 2:
        # Arguments provided: input_file output_file
        convert_csv_to_json(sys.argv[1], sys.argv[2])
    else:
        # No arguments: process all files in data/csv directory
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(script_dir)
            csv_dir = os.path.join(project_dir, "data", "csv")
            json_dir = os.path.join(project_dir, "data", "json")
            
            os.makedirs(json_dir, exist_ok=True)
            
            if not os.path.exists(csv_dir):
                print(f"Error: Input directory not found: {csv_dir}")
                return
                
            for filename in os.listdir(csv_dir):
                if filename.endswith(".csv"):
                    csv_path = os.path.join(csv_dir, filename)
                    json_path = os.path.join(json_dir, filename.replace(".csv", ".json"))
                    convert_csv_to_json(csv_path, json_path)
        except Exception as e:
            print(f"Error processing directory: {str(e)}")

if __name__ == "__main__":
    main()