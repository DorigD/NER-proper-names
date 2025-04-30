from scripts.converters.converter_txt import convert_txt_to_json
from scripts.converters.converter_csv import convert_csv_to_json
import os
SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)
txt_dir = os.path.join(PROJECT_DIR, "data", "txt")
json_dir = os.path.join(PROJECT_DIR, "data", "json")

def convert_file(input_file_path, output_file_path=os.path.join(json_dir, "result.json"), replace=True, cleanup=True):
    """
    Convert a file from one format to another based on its extension.
    
    Args:
        input_file_path (str): Path to the input file.
        output_file_path (str): Path to the output file.
    
    Returns:
        bool: True if conversion was successful, False otherwise.
    """
    if cleanup and os.path.exists(output_file_path):
        os.remove(output_file_path)
    try:
        if input_file_path.endswith(".txt"):
            return convert_txt_to_json(input_file_path, output_file_path, replace)
        elif input_file_path.endswith(".csv"):
            return convert_csv_to_json(input_file_path, output_file_path, replace)
        else:
            print(f"Unsupported file format: {input_file_path}")
            return False
    except Exception as e:
        print(f"Error converting {input_file_path}: {str(e)}")
        return False


