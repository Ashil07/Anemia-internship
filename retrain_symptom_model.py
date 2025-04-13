"""
retrain_symptom_model.py - Script to retrain the symptom-based anemia prediction model
with special handling for no-symptom cases
"""

import os
import sys
from src.symptom_prediction import run_symptom_model_training

if __name__ == "__main__":
    print("Retraining the symptom-based anemia prediction model...")
    output_dir = "./output/symptom_model_evaluation"
    model_path = "./models/symptom_anemia_prediction_model.pkl"
    
    os.makedirs(output_dir, exist_ok=True)

    run_symptom_model_training(output_dir, model_path)
    
    print("Model retraining complete!") 