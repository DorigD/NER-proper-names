import json
import os
import sys
import re
from utils.config import PERSON_TAG_PATTERN, TITLES, DATA_DIR
LABELS = {"O": 0, "B-PERSON": 1, "I-PERSON": 2, "TITLE": 3}

def convert_xml_to_json(xml_file_path, json_file_path, replace=True):
    dataset = []
    
    try:
        with open(xml_file_path, "r", encoding="utf-8") as file:
            content = file.read()
            
        # Process each document in the file
        docs = re.findall(r'<DOC>.*?</DOC>', content, re.DOTALL)
        
        for doc in docs:
            # Extract text portion only
            text_match = re.search(r'<TEXT>(.*?)</TEXT>', doc, re.DOTALL)
            if not text_match:
                continue
                
            text = text_match.group(1).strip()
            
            # Process text by paragraphs (treating each paragraph as a sentence for NER)
            paragraphs = re.split(r'\n\s*\n', text)
            
            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                    
                # Skip header lines that are likely not part of the main content
                if paragraph.startswith('\t   '):
                    paragraph = paragraph[4:]
                
                current_sentence = {"tokens": [], "ner_tags": []}
                
                # Keep track of our position as we parse
                pos = 0
                while pos < len(paragraph):
                    # Check for beginning of entity
                    b_enamex_match = re.search(r'<b_enamex type="([^"]+)"(?: status="[^"]+")?>', paragraph[pos:])
                    b_numex_match = re.search(r'<b_numex type="[^"]+">',  paragraph[pos:])
                    b_timex_match = re.search(r'<b_timex type="[^"]+"(?: alt="[^"]+")?>', paragraph[pos:])
                    
                    # If no entity tags are found, process the rest as plain text
                    if not (b_enamex_match or b_numex_match or b_timex_match):
                        # Process the rest of the text as normal tokens
                        text_part = paragraph[pos:]
                        tokens = tokenize_text(text_part)
                        for token in tokens:
                            if token.strip():  # Skip empty tokens
                                if token in TITLES:
                                    current_sentence["tokens"].append(token)
                                    if "TITLE" in LABELS:
                                        current_sentence["ner_tags"].append(LABELS["TITLE"])
                                    else:
                                        current_sentence["ner_tags"].append(LABELS["O"])
                                else:
                                    current_sentence["tokens"].append(token)
                                    current_sentence["ner_tags"].append(LABELS["O"])
                        break
                    
                    # Find the closest entity tag
                    tag_positions = [
                        (b_enamex_match.start(), "enamex", b_enamex_match.group(0), b_enamex_match.group(1))
                        if b_enamex_match else (float('inf'), None, None, None),
                        (b_numex_match.start(), "numex", b_numex_match.group(0), None)
                        if b_numex_match else (float('inf'), None, None, None),
                        (b_timex_match.start(), "timex", b_timex_match.group(0), None)
                        if b_timex_match else (float('inf'), None, None, None)
                    ]
                    
                    closest_tag = min(tag_positions, key=lambda x: x[0])
                    
                    if closest_tag[0] > 0:
                        # Process text before the entity
                        text_before = paragraph[pos:pos + closest_tag[0]]
                        tokens = tokenize_text(text_before)
                        for token in tokens:
                            if token.strip():  # Skip empty tokens
                                if token in TITLES:
                                    current_sentence["tokens"].append(token)
                                    if "TITLE" in LABELS:
                                        current_sentence["ner_tags"].append(LABELS["TITLE"])
                                    else:
                                        current_sentence["ner_tags"].append(LABELS["O"])
                                else:
                                    current_sentence["tokens"].append(token)
                                    current_sentence["ner_tags"].append(LABELS["O"])
                    
                    # Move position past the opening tag
                    pos += closest_tag[0] + len(closest_tag[2])
                    
                    # If it's an entity of interest (PERSON)
                    # Replace direct comparison with regex pattern matching
                    if closest_tag[1] == "enamex" and closest_tag[3] and PERSON_TAG_PATTERN.match(closest_tag[3]):
                        # Find the corresponding end tag
                        end_tag = f"<e_{closest_tag[1]}>"
                        end_pos = paragraph.find(end_tag, pos)
                        
                        if end_pos != -1:
                            # Extract entity text
                            entity_text = paragraph[pos:end_pos]
                            entity_tokens = tokenize_text(entity_text)
                            
                            # Add entity tokens with proper BIO tags
                            for i, token in enumerate(entity_tokens):
                                if token.strip():  # Skip empty tokens
                                    if token.lower() in TITLES:
                                        current_sentence["tokens"].append(token)
                                        if "TITLE" in LABELS:
                                            current_sentence["ner_tags"].append(LABELS["TITLE"])
                                        else:
                                            current_sentence["ner_tags"].append(LABELS["O"])
                                    else:
                                        current_sentence["tokens"].append(token)
                                        if i == 0:
                                            current_sentence["ner_tags"].append(LABELS["B-PERSON"])
                                        else:
                                            current_sentence["ner_tags"].append(LABELS["I-PERSON"])
                            
                            # Move past the entity and its end tag
                            pos = end_pos + len(end_tag)
                        else:
                            # Something went wrong, just move forward
                            pos += 1
                    else:
                        # For non-PERSON entities, find the end tag and skip it
                        end_tag = f"<e_{closest_tag[1]}>"
                        end_pos = paragraph.find(end_tag, pos)
                        
                        if end_pos != -1:
                            # Extract entity text but mark as non-entity
                            entity_text = paragraph[pos:end_pos]
                            entity_tokens = tokenize_text(entity_text)
                            
                            # Add tokens as non-entities
                            for token in entity_tokens:
                                if token.strip():  # Skip empty tokens
                                    current_sentence["tokens"].append(token)
                                    current_sentence["ner_tags"].append(LABELS["O"])
                            
                            # Move past the entity and its end tag
                            pos = end_pos + len(end_tag)
                        else:
                            # Something went wrong, just move forward
                            pos += 1
                
                # Add the sentence to the dataset if it's not empty
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
        print(f"{action} {xml_file_path} to {json_file_path}")
        
        return True
    except Exception as e:
        print(f"Error converting {xml_file_path}: {str(e)}")
        return False

def tokenize_text(text):
    """Simple tokenizer that splits on whitespace and punctuation."""
    # Replace punctuation with space + punctuation to ensure they're separate tokens
    for punct in '.,;:!?"()[]{}':
        text = text.replace(punct, f' {punct} ')
    
    # Split on whitespace and filter out empty tokens
    tokens = [token for token in text.split() if token.strip()]
    return tokens

def main():
    if len(sys.argv) < 3:
        print("Usage: python conveter_xml.py <input_xml_file> <output_json_file> [--append]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    replace = True
    
    if len(sys.argv) > 3 and sys.argv[3] == "--append":
        replace = False
    
    success = convert_xml_to_json(input_file, output_file, replace)
    if success:
        print(f"Successfully processed {input_file}")
    else:
        print(f"Failed to process {input_file}")



