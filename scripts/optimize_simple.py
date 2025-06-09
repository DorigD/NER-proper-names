"""
Simple Two-Phase Hyperparameter Optimization
===========================================

A simplified approach to two-phase optimization:
1. Phase 1: Broad exploration (15 trials)
2. Analyze results and generate focused ranges
3. Phase 2: Focused search (25 trials)

Usage:
    Edit the parameters below and run:
    python scripts/optimize_simple.py
"""

import optuna
import os
import sys
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
import pandas as pd
import json
from datetime import datetime
from pathlib import Path



# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train import main as train_model
from utils.config import MODELS_DIR, LOGS_DIR

MODEL_OUTPUT_DIR = os.path.join(MODELS_DIR, "roberta-finetuned-ner-hypertuned")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "optuna_tuning.log")),
        logging.StreamHandler()
    ]
)

def get_search_ranges(phase="phase1", focused_ranges=None):
    """Get parameter search ranges for different phases"""
    
    if phase == "phase1":
        # Phase 1: Broad exploration
        return {
            "num_attention_heads": (2, 8),
            "max_relative_position": (3, 10),
            "dropout": (0.1, 0.4),
            "crf_weight": (0.3, 0.9),
            "focal_weight": (0.05, 0.4),
            "dice_weight": (0.05, 0.5),
            "alpha": (0.1, 0.5),
            "gamma": (1.0, 3.0),
            "person_weight": (2.0, 8.0),
            "b_weight": (1.5, 5.0),
            "i_end_weight": (1.0, 4.0),
            "context_weight": (0.5, 3.0),
            "confidence_threshold": (0.5, 0.9),
            "learning_rate": (5e-6, 8e-5),
            "weight_decay": (0.001, 0.02),
            "warmup_ratio": (0.02, 0.2)
        }
    
    elif phase == "phase2" and focused_ranges:
        # Phase 2: Use focused ranges from analysis
        return focused_ranges
    
    else:
        # Default focused ranges if analysis not available
        return {
            "num_attention_heads": (4, 6),
            "max_relative_position": (5, 8),
            "dropout": (0.15, 0.3),
            "crf_weight": (0.5, 0.7),
            "focal_weight": (0.1, 0.25),
            "dice_weight": (0.15, 0.35),
            "alpha": (0.2, 0.35),
            "gamma": (1.8, 2.3),
            "person_weight": (4.0, 6.5),
            "b_weight": (2.5, 3.5),
            "i_end_weight": (2.0, 2.8),
            "context_weight": (1.3, 2.2),
            "confidence_threshold": (0.65, 0.8),
            "learning_rate": (1.5e-5, 3.5e-5),
            "weight_decay": (0.007, 0.013),
            "warmup_ratio": (0.08, 0.12)
        }

def objective(trial, phase="phase1", focused_ranges=None):
    """Optuna objective function for hyperparameter optimization"""
    
    # Get search ranges based on phase
    ranges = get_search_ranges(phase, focused_ranges)
    
    # Define search space using the ranges
    config = {
        # Model architecture hyperparameters
        "num_attention_heads": trial.suggest_int("num_attention_heads", *ranges["num_attention_heads"]),
        "max_relative_position": trial.suggest_int("max_relative_position", *ranges["max_relative_position"]),
        "dropout": trial.suggest_float("dropout", *ranges["dropout"]),
        
        # Loss weights
        "crf_weight": trial.suggest_float("crf_weight", *ranges["crf_weight"]),
        "focal_weight": trial.suggest_float("focal_weight", *ranges["focal_weight"]),
        "dice_weight": trial.suggest_float("dice_weight", *ranges["dice_weight"]),
        
        # Focal loss parameters
        "alpha": trial.suggest_float("alpha", *ranges["alpha"]),
        "gamma": trial.suggest_float("gamma", *ranges["gamma"]),
        "person_weight": trial.suggest_float("person_weight", *ranges["person_weight"]),
        
        # Dice loss parameters
        "b_weight": trial.suggest_float("b_weight", *ranges["b_weight"]),
        "i_end_weight": trial.suggest_float("i_end_weight", *ranges["i_end_weight"]),
        "context_weight": trial.suggest_float("context_weight", *ranges["context_weight"]),
        
        # Post-processing
        "confidence_threshold": trial.suggest_float("confidence_threshold", *ranges["confidence_threshold"]),
        
        # Training parameters
        "learning_rate": trial.suggest_float("learning_rate", *ranges["learning_rate"], log=True),
        "weight_decay": trial.suggest_float("weight_decay", *ranges["weight_decay"]),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16]),
        "warmup_ratio": trial.suggest_float("warmup_ratio", *ranges["warmup_ratio"]),
        "lr_scheduler": trial.suggest_categorical("lr_scheduler", ["linear", "cosine", "cosine_with_restarts"]),
        
        "skip_postprocessing_eval": True,
        "skip_augmentation": True
    }
    
    # Fixed parameters
    config["label_smoothing"] = 0.1
    config["gradient_accumulation"] = 2 if config["batch_size"] < 16 else 1
    config["epochs"] = 8 if phase == "phase1" else 10  # Faster for phase1
    
    # Log the current trial config
    logging.info(f"Trial {trial.number} ({phase}): {config}")
    
    # Create model-specific output directory for this trial
    trial_output_dir = os.path.join(MODEL_OUTPUT_DIR, f"trial_{trial.number}")
    os.makedirs(trial_output_dir, exist_ok=True)
    
    # Save trial config
    with open(os.path.join(trial_output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    try:
        # Run training
        results = train_model(
            config=config, 
            output_dir=trial_output_dir, 
            model_name="roberta-base",
            use_simplified_model=True,  # Use simplified model for faster trials
            include_title_tags=False
        )
        
        # Get F1 score
        person_f1 = results.get("person_f1", results.get("eval_person_f1", 0.0))
        
        logging.info(f"Trial {trial.number} achieved F1: {person_f1:.4f}")
        
        # Report for pruning
        trial.report(person_f1, step=1)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        return person_f1
    
    except Exception as e:
        logging.error(f"Trial {trial.number} failed: {str(e)}")
        return 0.0

def analyze_phase1_results(study_timestamp):
    """Analyze phase1 results and generate focused ranges for phase2"""
    study_dir = os.path.join(LOGS_DIR, f"optuna_study_{study_timestamp}")
    storage_url = f"sqlite:///{os.path.join(study_dir, 'optuna_study.db')}"
    
    try:
        # Load the study
        study = optuna.load_study(
            study_name=f"ner_person_optimization_{study_timestamp}",
            storage=storage_url
        )
        
        logging.info(f"Analyzing {len(study.trials)} trials from phase1")
        
        # Get top 30% of trials
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            logging.warning("No completed trials found!")
            return None
            
        sorted_trials = sorted(completed_trials, key=lambda x: x.value, reverse=True)
        top_trials = sorted_trials[:max(1, len(sorted_trials) // 3)]
        
        logging.info(f"Using top {len(top_trials)} trials for phase2 range analysis")
        
        # Calculate focused ranges
        focused_ranges = {}
        
        for param_name in top_trials[0].params.keys():
            if param_name in ["batch_size", "lr_scheduler"]:  # Skip categorical
                continue
                
            values = [trial.params[param_name] for trial in top_trials]
            mean_val = np.mean(values)
            std_val = np.std(values) if len(values) > 1 else 0.1 * mean_val
            
            # Create focused range: mean ± 1.5*std
            original_ranges = get_search_ranges("phase1")
            if param_name in original_ranges:
                orig_min, orig_max = original_ranges[param_name]
                
                new_min = max(orig_min, mean_val - 1.5 * std_val)
                new_max = min(orig_max, mean_val + 1.5 * std_val)
                
                # Ensure minimum range width
                if new_max - new_min < 0.1 * (orig_max - orig_min):
                    center = (new_min + new_max) / 2
                    width = 0.1 * (orig_max - orig_min)
                    new_min = max(orig_min, center - width/2)
                    new_max = min(orig_max, center + width/2)
                
                if param_name in ["num_attention_heads", "max_relative_position"]:
                    focused_ranges[param_name] = (int(new_min), int(new_max))
                else:
                    focused_ranges[param_name] = (new_min, new_max)
        
        # Save focused ranges
        ranges_file = os.path.join(study_dir, "phase2_ranges.json")
        with open(ranges_file, "w") as f:
            json.dump(focused_ranges, f, indent=2)
        
        logging.info(f"Phase2 focused ranges saved to {ranges_file}")
        logging.info(f"Best phase1 F1: {study.best_value:.4f}")
        
        return focused_ranges
        
    except Exception as e:
        logging.error(f"Error analyzing phase1 results: {e}")
        return None

def run_optimization(n_trials=15, continue_from=None, phase="phase1", focused_ranges=None):
    """Run Optuna hyperparameter optimization"""
    
    if continue_from:
        # Continue existing study
        study_dir = os.path.join(LOGS_DIR, f"optuna_study_{continue_from}")
        storage_url = f"sqlite:///{os.path.join(study_dir, 'optuna_study.db')}"
        study_name = f"ner_person_optimization_{continue_from}"
        
        if not os.path.exists(os.path.join(study_dir, 'optuna_study.db')):
            raise FileNotFoundError(f"Cannot find study database at {study_dir}")
            
        logging.info(f"Continuing optimization from existing study: {study_name} (Phase: {phase})")
        
    else:
        # Create new study
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_dir = os.path.join(LOGS_DIR, f"optuna_study_{timestamp}")
        Path(study_dir).mkdir(parents=True, exist_ok=True)
        storage_url = f"sqlite:///{os.path.join(study_dir, 'optuna_study.db')}"
        study_name = f"ner_person_optimization_{timestamp}"
        
        logging.info(f"Creating new optimization study: {study_name} (Phase: {phase})")
    
    # For phase2, try to load focused ranges if not provided
    if phase == "phase2" and focused_ranges is None:
        ranges_file = os.path.join(study_dir, "phase2_ranges.json")
        if os.path.exists(ranges_file):
            with open(ranges_file, "r") as f:
                focused_ranges = json.load(f)
            logging.info("Loaded focused ranges from phase1 analysis")
        else:
            logging.warning("No phase2 ranges found, using default focused ranges")
    
    # Configure pruner
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=3 if phase == "phase1" else 5,
        n_warmup_steps=1,
        interval_steps=1
    )
    
    # Create study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        pruner=pruner
    )
    
    # Log existing trials
    if continue_from and len(study.trials) > 0:
        logging.info(f"Loaded existing study with {len(study.trials)} completed trials")
        if study.best_trial:
            logging.info(f"Current best F1: {study.best_value:.4f} from trial {study.best_trial.number}")
    
    # Run optimization
    study.optimize(lambda trial: objective(trial, phase, focused_ranges), n_trials=n_trials)
    
    # Results
    best_params = study.best_params
    best_value = study.best_value
    
    logging.info(f"\nBest trial: {study.best_trial.number}")
    logging.info(f"Best F1 score: {best_value:.4f}")
    logging.info(f"Best hyperparameters: {best_params}")
    
    # Save best parameters
    with open(os.path.join(study_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)
    
    # Create plots
    try:
        # Parameter importance
        param_importance = optuna.importance.get_param_importances(study)
        importance_df = pd.DataFrame(
            list(param_importance.items()),
            columns=['Parameter', 'Importance']
        ).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Parameter'], importance_df['Importance'])
        plt.xlabel('Importance')
        plt.title(f'Hyperparameter Importance ({phase})')
        plt.tight_layout()
        plt.savefig(os.path.join(study_dir, f'parameter_importance_{phase}.png'))
        plt.close()
        
        # Optimization history
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig(os.path.join(study_dir, f'optimization_history_{phase}.png'))
        plt.close()
        
    except Exception as e:
        logging.error(f"Error creating plots: {str(e)}")
    
    return best_params, best_value, study

def run_two_phase_optimization(continue_from_timestamp=None):
    """Run the complete two-phase optimization pipeline"""
    
    print("🚀 Starting Two-Phase Hyperparameter Optimization")
    print("=" * 60)
    
    # Check if we should resume from an existing study
    if continue_from_timestamp:
        study_dir = os.path.join(LOGS_DIR, f"optuna_study_{continue_from_timestamp}")
        if os.path.exists(os.path.join(study_dir, 'optuna_study.db')):
            print(f"📂 Found existing study: {continue_from_timestamp}")
            
            # Load existing study to check progress
            storage_url = f"sqlite:///{os.path.join(study_dir, 'optuna_study.db')}"
            try:
                study = optuna.load_study(
                    study_name=f"ner_person_optimization_{continue_from_timestamp}",
                    storage=storage_url
                )
                
                completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                print(f"📊 Existing study has {len(completed_trials)} completed trials")
                
                # Check if Phase 1 is complete (assume Phase 1 if < 20 trials)
                if len(completed_trials) < 20:
                    print("🔄 Resuming Phase 1...")
                    remaining_trials = 15 - len(completed_trials)
                    if remaining_trials > 0:
                        best_params_1, best_f1_1, study_1 = run_optimization(
                            n_trials=remaining_trials,
                            continue_from=continue_from_timestamp,
                            phase="phase1"
                        )
                    else:
                        best_params_1, best_f1_1, study_1 = study.best_params, study.best_value, study
                    
                    timestamp = continue_from_timestamp
                    print(f"✅ Phase 1 completed! Best F1: {best_f1_1:.4f}")
                    
                else:
                    print("✅ Phase 1 already completed")
                    best_params_1, best_f1_1 = study.best_params, study.best_value
                    timestamp = continue_from_timestamp
                
            except Exception as e:
                print(f"❌ Error loading existing study: {e}")
                print("🔄 Starting fresh Phase 1...")
                best_params_1, best_f1_1, study_1 = run_optimization(n_trials=15, phase="phase1")
                timestamp = study_1.study_name.split('_')[-1]
        else:
            print(f"❌ Study {continue_from_timestamp} not found, starting fresh...")
            best_params_1, best_f1_1, study_1 = run_optimization(n_trials=15, phase="phase1")
            timestamp = study_1.study_name.split('_')[-1]
    else:
        # Fresh start - Phase 1
        print("\n📊 PHASE 1: Broad Exploration")
        print("Running 15 trials with broad parameter ranges...")
        
        best_params_1, best_f1_1, study_1 = run_optimization(
            n_trials=15, 
            phase="phase1"
        )
        
        # Extract timestamp from study
        timestamp = study_1.study_name.split('_')[-1]
        print(f"\n✅ Phase 1 completed! Best F1: {best_f1_1:.4f}")
    
    print(f"📁 Results saved to: logs/optuna_study_{timestamp}/")
    
    # Analyze Phase 1 results
    print("\n🔍 ANALYZING PHASE 1 RESULTS...")
    focused_ranges = analyze_phase1_results(timestamp)
    
    if focused_ranges:
        print("✅ Phase 1 analysis completed - focused ranges generated")
        
        # Phase 2: Focused search
        print("\n🎯 PHASE 2: Focused Search")
        print("Running 25 trials with focused parameter ranges...")
        
        best_params_2, best_f1_2, study_2 = run_optimization(
            n_trials=25,
            continue_from=timestamp,
            phase="phase2",
            focused_ranges=focused_ranges
        )
        
        print(f"\n✅ Phase 2 completed! Best F1: {best_f1_2:.4f}")
        
        # Summary
        print(f"\n📈 OPTIMIZATION SUMMARY:")
        print(f"   Phase 1 Best F1: {best_f1_1:.4f}")
        print(f"   Phase 2 Best F1: {best_f1_2:.4f}")
        print(f"   Improvement: {((best_f1_2 - best_f1_1) / best_f1_1 * 100):+.2f}%")
        print(f"\n🎉 Two-phase optimization completed!")
        print(f"📁 All results in: logs/optuna_study_{timestamp}/")
        print(f"\n💡 To train final model with best parameters:")
        print(f"   python scripts/train.py --config logs/optuna_study_{timestamp}/best_params.json")
        
        return best_params_2, best_f1_2
    
    else:
        print("❌ Phase 1 analysis failed - cannot proceed to Phase 2")
        return best_params_1, best_f1_1

# =============================================================================
# CONFIGURATION PARAMETERS - Edit these to control the optimization
# =============================================================================

# Which phase to run: "phase1", "phase2", or "both"
PHASE = "both"

# Number of trials (None = use defaults: 15 for phase1, 25 for phase2)
N_TRIALS = None

# For phase2 or resuming: timestamp to continue from (e.g., "20250607_143022")
# - If PHASE="both" and CONTINUE_FROM is set, it will resume the study
# - If PHASE="phase2", CONTINUE_FROM is required
# - If PHASE="phase1", CONTINUE_FROM is ignored
CONTINUE_FROM = "20250607_192744"

# =============================================================================


if __name__ == "__main__":
    # Check for CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
      # Set defaults
    phase = PHASE
    trials = N_TRIALS
    continue_from = CONTINUE_FROM
    
    if phase == "both":
        # Run complete two-phase pipeline
        print("🚀 Running complete two-phase optimization pipeline")
        # Pass the continue_from timestamp if provided
        run_two_phase_optimization(continue_from_timestamp=continue_from)
        
    elif phase == "phase1":
        # Run Phase 1 only
        n_trials = trials or 15
        print(f"🚀 Running Phase 1 with {n_trials} trials")
        best_params, best_f1, study = run_optimization(n_trials=n_trials, phase="phase1")
        timestamp = study.study_name.split('_')[-1]
        print(f"✅ Phase 1 completed! Best F1: {best_f1:.4f}")
        print(f"📁 Results saved to: logs/optuna_study_{timestamp}/")
        print(f"\n💡 To run Phase 2, edit these parameters at the top of the script:")
        print(f"   PHASE = 'phase2'")
        print(f"   CONTINUE_FROM = '{timestamp}'")
        
    elif phase == "phase2":
        # Run Phase 2 only
        if not continue_from:
            print("❌ Error: CONTINUE_FROM timestamp required for phase2")
            print("   Set CONTINUE_FROM = 'YYYYMMDD_HHMMSS' from your Phase 1 run")
            sys.exit(1)
            
        # Analyze Phase 1 first
        print("🔍 Analyzing Phase 1 results...")
        focused_ranges = analyze_phase1_results(continue_from)
        
        if focused_ranges:
            n_trials = trials or 25
            print(f"🎯 Running Phase 2 with {n_trials} trials")
            best_params, best_f1, study = run_optimization(
                n_trials=n_trials,
                continue_from=continue_from,
                phase="phase2",
                focused_ranges=focused_ranges
            )
            print(f"✅ Phase 2 completed! Best F1: {best_f1:.4f}")
            print(f"📁 Results saved to: logs/optuna_study_{continue_from}/")
            print(f"\n💡 To train final model:")
            print(f"   python scripts/train.py --config logs/optuna_study_{continue_from}/best_params.json")
        else:
            print("❌ Phase 1 analysis failed - cannot run Phase 2")
            sys.exit(1)
    
    else:
        print(f"❌ Error: Invalid phase '{phase}'. Use 'both', 'phase1', or 'phase2'")
        sys.exit(1)
