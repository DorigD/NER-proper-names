from main import NER
from utils.config import DATA_DIR, PROJECT_DIR 
from datasets import load_from_disk
import os
import multiprocessing
import json
import subprocess
import sys

def run_training(data):
    ner = NER()
    used_dir = os.path.join(DATA_DIR, "used")
    os.makedirs(used_dir, exist_ok=True)
    
    # Define the path to training results
    training_results_path = os.path.join(PROJECT_DIR, "logs", "training_results.json")
    
    for file in os.listdir(data):
        try:
            model_path = ner.train(training_data=os.path.join(data, file), save_train=file)
            if model_path:
                ner.load_model(model_path)
                ner.set_model(model_path)
                print(f"------------Model updated to: {model_path} ----------------------")
                
                # Update the training_results.json to include the filename
                if os.path.exists(training_results_path):
                    try:
                        with open(training_results_path, "r") as f:
                            results = json.load(f)
                        
                        # Add filename to the last entry
                        if results and isinstance(results, list) and len(results) > 0:
                            results[-1]["dataset_filename"] = file
                            
                            # Write the updated results back to the file
                            with open(training_results_path, "w") as f:
                                json.dump(results, f, indent=2)
                            
                            print(f"Added dataset filename '{file}' to training results")
                    except Exception as e:
                        print(f"Error updating training results: {str(e)}")
                
                # Move the processed file to used directory
                os.rename(os.path.join(data, file), os.path.join(used_dir, file))
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")

if __name__ == '__main__':

    multiprocessing.freeze_support()
    data = os.path.join(DATA_DIR, "raw", "general")
    run_training(data=data)
    data = os.path.join(DATA_DIR, "raw", "Social")
    run_training(data=data)
    