"""
NER Training Pipeline
===================

This script provides a complete pipeline for training NER models using optimized hyperparameters
from Optuna optimization studies.

Configuration:
--------------
Simply modify the variables in the CONFIGURATION section below to customize your training:

- OPTUNA_STUDY_TIMESTAMP: Set to your Optuna study timestamp
- USE_SIMPLIFIED_MODEL: True for faster training, False for best performance
- TRAINING_EPOCHS: Number of epochs to train (15 recommended for final models)
- TEST_MODE: True for quick testing with reduced epochs
- INCLUDE_TITLE_TAGS: True to include B-TITLE/I-TITLE tags, False for PERSON-only NER

Usage Examples:
---------------
1. Basic usage (uses current configuration):
   python main.py

2. Run both pipelines (with and without TITLE tags):
   # In Python interpreter or script:
   from main import run_both_pipelines
   run_both_pipelines()

3. Run with custom title setting:
   # In Python interpreter or script:
   from main import run_pipeline_with_title_settings
   run_pipeline_with_title_settings(include_title_tags=False)  # NO-TITLE only
   run_pipeline_with_title_settings(include_title_tags=True)   # With TITLE tags

Features:
---------
- Automatic environment validation
- Configuration consistency checking across all modules
- Support for both TITLE and NO-TITLE datasets
- Integration with Optuna hyperparameter optimization results
- Flexible model selection (simplified vs complex)
- Comprehensive error handling and logging
- Dynamic label configuration management

Requirements:
------------
- Completed Optuna hyperparameter optimization study
- Raw data files in data/raw/Social/ and data/raw/general/
- Valid dataset structure and converters
"""

import os
import json
from scripts.preprocess import create_ner_dataset
from scripts.converters.Converter import convert_file
from utils.config import PROJECT_DIR, DATA_DIR, MODELS_DIR
from utils.label_config import get_label_config, save_label_config
from scripts.train import main as train_model

# ============================================================================
# CONFIGURATION - Modify these variables as needed
# ============================================================================

# Optuna study timestamp (update this to match your study)
OPTUNA_STUDY_TIMESTAMP = "20250607_192744"

# Model type: False for complex model (best performance), True for simplified model (faster)
USE_SIMPLIFIED_MODEL = False

# Number of training epochs (15 recommended for final models, 8-10 for testing)
TRAINING_EPOCHS = 15

# Test mode: True for quick testing (reduces epochs to 3), False for full training
TEST_MODE = False

# Include TITLE tags: True to include B-TITLE/I-TITLE tags, False for PERSON-only NER
INCLUDE_TITLE_TAGS = False

# ============================================================================

input_file_social = os.path.join(DATA_DIR, "raw", "Social")
input_file_general = os.path.join(DATA_DIR, "raw", "general")

def load_best_hyperparameters(study_timestamp):
    """Load the best hyperparameters from Optuna optimization"""
    best_params_path = os.path.join(PROJECT_DIR, "logs", f"optuna_study_{study_timestamp}", "best_params.json")
    
    try:
        with open(best_params_path, 'r') as f:
            config = json.load(f)
        print(f"✅ Loaded optimized hyperparameters from: {best_params_path}")
        return config
    except FileNotFoundError:
        print(f"⚠️  Best parameters file not found: {best_params_path}")
        print("Using default hyperparameters...")
        return None

def validate_environment(study_timestamp):
    """Validate that required directories and files exist"""
    issues = []
    
    # Check if raw data directories exist
    if not os.path.exists(input_file_social):
        issues.append(f"Social data directory not found: {input_file_social}")
    elif not os.listdir(input_file_social):
        issues.append(f"Social data directory is empty: {input_file_social}")
    
    if not os.path.exists(input_file_general):
        issues.append(f"General data directory not found: {input_file_general}")
    elif not os.listdir(input_file_general):
        issues.append(f"General data directory is empty: {input_file_general}")
    
    # Check if Optuna results exist
    optuna_path = os.path.join(PROJECT_DIR, "logs", f"optuna_study_{study_timestamp}")
    if not os.path.exists(optuna_path):
        issues.append(f"Optuna study directory not found: {optuna_path}")
    
    best_params_path = os.path.join(optuna_path, "best_params.json")
    if not os.path.exists(best_params_path):
        issues.append(f"Best parameters file not found: {best_params_path}")
    
    # Create required directories
    os.makedirs(os.path.join(DATA_DIR, "ds", "NO-TITLE"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "ds", "TITLE"), exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    return issues

def process_and_train_dataset(raw_data_dir, dataset_name, study_timestamp, use_simplified_model, epochs, include_title_tags):
    """Process raw data and train model for a specific dataset"""
    print(f"\n{'='*60}")
    print(f"🔄 Processing dataset: {dataset_name}")
    print(f"📁 Source directory: {raw_data_dir}")
    print(f"{'='*60}")
    
    # Load optimized hyperparameters
    config = load_best_hyperparameters(study_timestamp)
    
    # Override config for final training with complex model
    if config:
        config["epochs"] = epochs
        # Remove optimization-specific parameters that don't apply to final training
        config.pop("skip_postprocessing_eval", None)
        config.pop("skip_augmentation", None)
    
    version = 1  # Start from version 1 for trained models
    
    for file in os.listdir(raw_data_dir):
        if not file.endswith(('.txt', '.json')):  # Skip non-data files
            continue
            
        input_file = os.path.join(raw_data_dir, file)
        dataset_id = file.split('.')[0]
        
        print(f"\n📄 Processing file: {file}")
        print(f"🔄 Converting to NER format...")
        
        # Convert file to NER format
        convert_file(input_file)
          # Check if conversion was successful
        result_json_path = os.path.join(DATA_DIR, "json", "result.json")
        if not os.path.exists(result_json_path):
            print(f"❌ Conversion failed for {file}, skipping...")
            continue
          # Determine if we should include title tags (use the flag parameter)
        model_suffix = "TITLE" if include_title_tags else "NO-TITLE"
        
        # Create dataset output directory
        dataset_output_dir = os.path.join(DATA_DIR, "ds", model_suffix, dataset_id)
        
        print(f"🏗️  Creating NER dataset with{'out' if not include_title_tags else ''} TITLE tags...")
        
        # Create NER dataset - this will save both to tokenized_train and ds directory
        create_ner_dataset(
            file_path=result_json_path,
            output_dir=None,  # This will default to tokenized_train
            include_title_tags=include_title_tags,
            save_test_separately=True,  # This will save copy to ds directory
            test_output_dir=dataset_output_dir  # Specify the ds directory path
        )
        
        # The training script expects data in tokenized_train, so use that path
        training_dataset_path = os.path.join(DATA_DIR, "tokenized_train")
        
        # Verify dataset was created
        if not os.path.exists(training_dataset_path):
            print(f"❌ Dataset creation failed for {dataset_id}, skipping training...")
            continue
        
        # Set up model paths
        model_name = "roberta-base"  # Always start from base model
        output_dir = os.path.join(MODELS_DIR, f"roberta-finetuned-ner-{model_suffix}-v{version}")
          # Save label configuration for this model
        save_label_config(output_dir, include_title_tags)
        
        print(f"\n🚀 Starting model training...")
        print(f"📊 Dataset: {training_dataset_path}")  # Updated to show correct path
        print(f"🤖 Model type: {'Complex' if not use_simplified_model else 'Simplified'}")
        print(f"🏷️  Include TITLE tags: {include_title_tags}")
        print(f"💾 Output directory: {output_dir}")
        
        try:
            # Train the model using the standardized dataset path
            results = train_model(
                config=config,
                test_mode=False,
                output_dir=output_dir,
                dataset_path=training_dataset_path,  # Use the standardized path
                model_name=model_name,
                use_simplified_model=use_simplified_model,
                include_title_tags=include_title_tags
            )
            
            print(f"✅ Training completed successfully!")
            if results and isinstance(results, dict):
                person_f1 = results.get("person_f1", results.get("eval_person_f1", "N/A"))
                print(f"📈 Person F1 Score: {person_f1}")
            
            version += 1
            
        except Exception as e:
            print(f"❌ Training failed for {dataset_id}: {str(e)}")
            continue
    
    print(f"\n✅ Completed processing {dataset_name} dataset")

def run_full_pipeline():
    """Run the complete data processing and training pipeline"""
    print("🚀 Starting NER Training Pipeline")
    print("=" * 60)
      # Apply test mode adjustment
    epochs = 3 if TEST_MODE else TRAINING_EPOCHS
    
    print(f"📋 Configuration:")
    print(f"   • Optuna Study: {OPTUNA_STUDY_TIMESTAMP}")
    print(f"   • Model Type: {'Complex' if not USE_SIMPLIFIED_MODEL else 'Simplified'}")
    print(f"   • Training Epochs: {epochs}")
    print(f"   • Include TITLE tags: {INCLUDE_TITLE_TAGS}")
    if TEST_MODE:
        print(f"   • TEST MODE: Enabled (reduced epochs)")
    print("=" * 60)
      # Validate environment
    print("🔍 Validating environment...")
    env_issues = validate_environment(OPTUNA_STUDY_TIMESTAMP)
    if env_issues:
        print("❌ Environment validation failed:")
        for issue in env_issues:
            print(f"   • {issue}")
        print("\n💡 Please ensure all required files and directories are present before running the pipeline.")
        return
    
    print("✅ Environment validation passed")
    
    # Validate configuration consistency
    validate_configuration_consistency()
      # Process Social dataset (typically without TITLE tags)
    if os.path.exists(input_file_social) and os.listdir(input_file_social):
        process_and_train_dataset(input_file_social, "Social", OPTUNA_STUDY_TIMESTAMP, USE_SIMPLIFIED_MODEL, epochs, INCLUDE_TITLE_TAGS)
    else:
        print(f"⚠️  Social dataset directory not found or empty: {input_file_social}")
    
    # Process General dataset (may include TITLE tags)
    if os.path.exists(input_file_general) and os.listdir(input_file_general):
        process_and_train_dataset(input_file_general, "General", OPTUNA_STUDY_TIMESTAMP, USE_SIMPLIFIED_MODEL, epochs, INCLUDE_TITLE_TAGS)
    else:
        print(f"⚠️  General dataset directory not found or empty: {input_file_general}")
    
    print("\n🎉 Pipeline completed successfully!")
    print("📁 Check the models/ directory for trained models")
    print("📊 Check the logs/ directory for training logs")

def run_pipeline_with_title_settings(include_title_tags=None):
    """
    Run the pipeline with custom title tag settings.
    This allows you to override the global INCLUDE_TITLE_TAGS setting for a single run.
    
    Args:
        include_title_tags: True/False to override global setting, None to use global setting
    """
    global INCLUDE_TITLE_TAGS
    
    # Temporarily override the global setting if specified
    original_setting = INCLUDE_TITLE_TAGS
    if include_title_tags is not None:
        INCLUDE_TITLE_TAGS = include_title_tags
    
    try:
        # Run the full pipeline
        run_full_pipeline()
    finally:
        # Restore original setting
        INCLUDE_TITLE_TAGS = original_setting

def run_both_pipelines():
    """
    Convenience function to run both pipelines - one with TITLE tags and one without.
    This is useful for comparing model performance with different tag schemas.
    """
    print("🎯 Running BOTH pipelines - with and without TITLE tags")
    print("=" * 60)
    
    print("\n🚀 FIRST RUN: Training with PERSON tags only (NO-TITLE)")
    print("=" * 60)
    run_pipeline_with_title_settings(include_title_tags=False)
    
    print("\n" + "=" * 80)
    print("🚀 SECOND RUN: Training with PERSON and TITLE tags")
    print("=" * 60)
    run_pipeline_with_title_settings(include_title_tags=True)
    
    print("\n🎉 Both pipelines completed!")
    print("📁 Check the models/ directory for trained models:")
    print("   • Models with 'NO-TITLE' are trained without TITLE tags")
    print("   • Models with 'TITLE' are trained with TITLE tags")

def validate_configuration_consistency():
    """
    Validate that all configuration files are properly coordinated.
    This function checks for potential conflicts between different configuration sources.
    """
    print("🔍 Validating configuration consistency...")
    
    # Import the label configurations
    from utils.label_config import get_label_config, LABELS_WITH_TITLE, LABELS_WITHOUT_TITLE
    from utils.config import LABELS as CONFIG_LABELS
    from scripts.preprocess import OUTPUT_TAGS_WITH_TITLE, OUTPUT_TAGS_PERSON_ONLY
    
    warnings = []
    
    # Check if config.py and label_config.py are consistent
    title_config = get_label_config(include_title=True)
    no_title_config = get_label_config(include_title=False)
    
    # Check label mappings consistency
    if LABELS_WITH_TITLE != OUTPUT_TAGS_WITH_TITLE:
        warnings.append("⚠️  LABELS_WITH_TITLE in label_config.py doesn't match OUTPUT_TAGS_WITH_TITLE in preprocess.py")
    
    if LABELS_WITHOUT_TITLE != OUTPUT_TAGS_PERSON_ONLY:
        warnings.append("⚠️  LABELS_WITHOUT_TITLE in label_config.py doesn't match OUTPUT_TAGS_PERSON_ONLY in preprocess.py")
    
    # Check if config.py hardcoded labels match any of our dynamic configs
    if CONFIG_LABELS == LABELS_WITH_TITLE:
        print("✅ config.py LABELS matches TITLE configuration")
    elif CONFIG_LABELS == LABELS_WITHOUT_TITLE:
        print("✅ config.py LABELS matches NO-TITLE configuration")
    else:
        warnings.append("⚠️  config.py LABELS doesn't match either dynamic configuration")
    
    # Validate current configuration
    current_config = get_label_config(INCLUDE_TITLE_TAGS)
    print(f"✅ Current configuration: {'TITLE' if INCLUDE_TITLE_TAGS else 'NO-TITLE'}")
    print(f"   Labels: {list(current_config['labels'].keys())}")
    print(f"   Number of labels: {current_config['num_labels']}")
    
    # Report warnings
    if warnings:
        print("\n⚠️  Configuration warnings found:")
        for warning in warnings:
            print(f"   {warning}")
        print("\n💡 Note: train.py handles these dynamically, but consistency is recommended.")
    else:
        print("✅ All configurations are consistent!")
    
    return len(warnings) == 0

if __name__ == "__main__":
    # Run the pipeline with current configuration
    # To run both pipelines, call run_both_pipelines() instead
    # run_full_pipeline()

    # Uncomment this line to run both pipelines automatically:
    run_both_pipelines()