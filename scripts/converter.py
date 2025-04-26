import json
import os
import sys
import re

# Define valid labels (Only keeping PERSON tags)
LABELS = {"O": 0, "B-PERSON": 1, "I-PERSON": 2} #add Tag for name Titles

# List of titles to remove
TITLES = {"Mr.", "Mrs.", "Miss", "Ms.", "Dr.", "Prof.", "Sir", "Madam",
          "President", "Chancellor", "Minister", "Mayor", "King", "Queen", "Pope"}

def convert_txt_to_json(txt_file_path, json_file_path):
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
                
                # Skip titles
                if word in TITLES:
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
                
                # Debug print to verify tokens and tags
                print(f"Word: {word}, Tag: {tag}")
                                
                # Convert NER tags to integer labels
                ner_label = LABELS.get(tag, LABELS["O"])

                current_sentence["tokens"].append(word)
                current_sentence["ner_tags"].append(ner_label)

        # Save last sentence
        if current_sentence["tokens"]:
            dataset.append(current_sentence)

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
        
        # Save to JSON
        with open(json_file_path, "w", encoding="utf-8") as out_file:
            json.dump(dataset, out_file, indent=4, ensure_ascii=False)

        print(f"Converted '{txt_file_path}' to '{json_file_path}'")
        print(f"  - Total lines processed: {line_count}")
        print(f"  - Lines skipped: {skipped_lines}")
        print(f"  - Sentences created: {len(dataset)}")
        return True
    except Exception as e:
        print(f"Error converting {txt_file_path}: {str(e)}")
        return False

# Add main function to process files
def main():
    if len(sys.argv) > 2:
        # Arguments provided: input_file output_file
        convert_txt_to_json(sys.argv[1], sys.argv[2])
    else:
        # No arguments: process all files in data/txt directory
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(script_dir)
            txt_dir = os.path.join(project_dir, "data", "txt")
            json_dir = os.path.join(project_dir, "data", "json")
            
            os.makedirs(json_dir, exist_ok=True)
            
            if not os.path.exists(txt_dir):
                print(f"Error: Input directory not found: {txt_dir}")
                return
                
            for filename in os.listdir(txt_dir):
                if filename.endswith(".txt"):
                    txt_path = os.path.join(txt_dir, filename)
                    json_path = os.path.join(json_dir, filename.replace(".txt", ".json"))
                    convert_txt_to_json(txt_path, json_path)
        except Exception as e:
            print(f"Error processing directory: {str(e)}")

if __name__ == "__main__":
    main()