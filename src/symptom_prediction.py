"""
symptom_prediction.py - Train and evaluate machine learning models for symptom-based anemia prediction

This script creates a synthetic dataset based on symptom and risk factor patterns,
then trains multiple machine learning models to predict anemia based on these features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to allow importing from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Machine learning libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                            roc_curve, auc, precision_recall_curve, f1_score)
from sklearn.pipeline import Pipeline

# Set random state for reproducibility
RANDOM_STATE = 42

def generate_symptom_data(n_samples=2000):
    """
    Generate synthetic dataset for symptom-based anemia prediction
    with more balanced representation of no-symptom cases
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame: Synthetic dataset with symptoms, risk factors, and anemia status
    """
    np.random.seed(RANDOM_STATE)
    
    anemia_status = np.random.choice([1, 0], size=n_samples, p=[0.6, 0.4])
    gender = np.random.choice([0, 1], size=n_samples)
    age = np.random.randint(18, 90, size=n_samples)
    
    # Ensure at least 15% of samples are no-symptom cases (all non-anemic)
    no_symptom_mask = np.random.random(n_samples) < 0.15
    anemia_status[no_symptom_mask] = 0
    
    # Initialize all symptom arrays to zero
    symptom_arrays = {
        'fatigue': np.zeros(n_samples, dtype=int),
        'shortness_breath': np.zeros(n_samples, dtype=int),
        'dizziness': np.zeros(n_samples, dtype=int),
        'pale_skin': np.zeros(n_samples, dtype=int),
        'heart_racing': np.zeros(n_samples, dtype=int),
        'headaches': np.zeros(n_samples, dtype=int),
        'cold_hands_feet': np.zeros(n_samples, dtype=int),
        'brittle_nails': np.zeros(n_samples, dtype=int),
        'poor_concentration': np.zeros(n_samples, dtype=int),
        'heavy_periods': np.zeros(n_samples, dtype=int),
        'recent_blood_loss': np.zeros(n_samples, dtype=int),
        'vegetarian_diet': np.zeros(n_samples, dtype=int),
        'gi_disorders': np.zeros(n_samples, dtype=int),
        'chronic_disease': np.zeros(n_samples, dtype=int)
    }
    
    # Set symptoms only for non-"no symptom" cases
    symptom_mask = ~no_symptom_mask
    
    # Define symptom probabilities for anemic vs non-anemic cases
    symptom_probs = {
        'fatigue': (0.85, 0.25),  # (anemic_prob, non_anemic_prob)
        'shortness_breath': (0.75, 0.15),
        'dizziness': (0.7, 0.2),
        'pale_skin': (0.8, 0.1),
        'heart_racing': (0.65, 0.15),
        'headaches': (0.6, 0.25),
        'cold_hands_feet': (0.55, 0.15),
        'brittle_nails': (0.45, 0.05),
        'poor_concentration': (0.5, 0.2),
        'recent_blood_loss': (0.45, 0.05),
        'vegetarian_diet': (0.35, 0.15),
        'gi_disorders': (0.4, 0.1),
        'chronic_disease': (0.5, 0.15)
    }
    
    # Generate symptoms based on anemia status for symptomatic cases
    for symptom, (anemic_prob, non_anemic_prob) in symptom_probs.items():
        for i in np.where(symptom_mask)[0]:
            if anemia_status[i] == 1:
                symptom_arrays[symptom][i] = np.random.binomial(1, anemic_prob)
            else:
                symptom_arrays[symptom][i] = np.random.binomial(1, non_anemic_prob)
    
    # Handle heavy periods separately (female only)
    female_mask = (gender == 1) & symptom_mask
    for i in np.where(female_mask)[0]:
        if anemia_status[i] == 1:
            symptom_arrays['heavy_periods'][i] = np.random.binomial(1, 0.65)
        else:
            symptom_arrays['heavy_periods'][i] = np.random.binomial(1, 0.15)
    
    # Create correlation between certain symptoms
    # Fatigue and shortness of breath often occur together
    for i in np.where(symptom_mask & (symptom_arrays['fatigue'] == 1))[0]:
        if symptom_arrays['shortness_breath'][i] == 0:
            symptom_arrays['shortness_breath'][i] = np.random.binomial(1, 0.7)
    
    # Pale skin and fatigue often occur together in anemic patients
    for i in np.where(symptom_mask & (anemia_status == 1) & (symptom_arrays['pale_skin'] == 1))[0]:
        if symptom_arrays['fatigue'][i] == 0:
            symptom_arrays['fatigue'][i] = np.random.binomial(1, 0.8)
    
    # Create DataFrame
    data = {
        'Gender': gender,
        'Age': age,
        'Fatigue': symptom_arrays['fatigue'],
        'Shortness_of_Breath': symptom_arrays['shortness_breath'],
        'Dizziness': symptom_arrays['dizziness'],
        'Pale_Skin': symptom_arrays['pale_skin'],
        'Heart_Racing': symptom_arrays['heart_racing'],
        'Headaches': symptom_arrays['headaches'],
        'Cold_Hands_Feet': symptom_arrays['cold_hands_feet'],
        'Brittle_Nails': symptom_arrays['brittle_nails'],
        'Poor_Concentration': symptom_arrays['poor_concentration'],
        'Heavy_Periods': symptom_arrays['heavy_periods'],
        'Recent_Blood_Loss': symptom_arrays['recent_blood_loss'],
        'Vegetarian_Diet': symptom_arrays['vegetarian_diet'],
        'GI_Disorders': symptom_arrays['gi_disorders'],
        'Chronic_Disease': symptom_arrays['chronic_disease'],
        'Anemia_Status': anemia_status
    }
    
    return pd.DataFrame(data)

def explore_symptom_data(symptom_df, output_dir=None):
    """
    Explore the symptom dataset with visualizations
    
    Args:
        symptom_df: DataFrame with symptom data
        output_dir: Directory to save visualizations
    """
    print("Dataset shape:", symptom_df.shape)
    print("\nClass distribution:")
    print(symptom_df['Anemia_Status'].value_counts())
    print(symptom_df['Anemia_Status'].value_counts(normalize=True) * 100)
    
    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    symptom_cols = [col for col in symptom_df.columns if col not in ['Gender', 'Age', 'Anemia_Status']]
    
    symptom_prevalence = pd.DataFrame()
    for symptom in symptom_cols:
        anemic_prevalence = symptom_df[symptom_df['Anemia_Status'] == 1][symptom].mean() * 100
        non_anemic_prevalence = symptom_df[symptom_df['Anemia_Status'] == 0][symptom].mean() * 100
        symptom_prevalence = pd.concat([symptom_prevalence, 
                                      pd.DataFrame({'Symptom': [symptom],
                                                  'Anemic': [anemic_prevalence],
                                                  'Non-Anemic': [non_anemic_prevalence]})], 
                                     ignore_index=True)
    
    # Plot the prevalence comparison
    plt.figure(figsize=(12, 8))
    melted_data = pd.melt(symptom_prevalence, id_vars=['Symptom'], value_vars=['Anemic', 'Non-Anemic'],
                         var_name='Status', value_name='Prevalence (%)')
    sns.barplot(x='Symptom', y='Prevalence (%)', hue='Status', data=melted_data)
    plt.xticks(rotation=45, ha='right')
    plt.title('Symptom Prevalence by Anemia Status')
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'symptom_prevalence.png'))
    else:
        plt.show()
    
    # Analyze correlations between symptoms and anemia status
    plt.figure(figsize=(10, 8))
    correlation_with_anemia = symptom_df.drop('Anemia_Status', axis=1).corrwith(symptom_df['Anemia_Status']).sort_values(ascending=False)
    sns.barplot(x=correlation_with_anemia.values, y=correlation_with_anemia.index)
    plt.title('Correlation of Features with Anemia Status')
    plt.xlabel('Correlation Coefficient')
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'feature_correlation.png'))
    else:
        plt.show()
    
    return correlation_with_anemia

def train_and_evaluate_models(X_train, X_test, y_train, y_test, output_dir=None):
    """
    Train and evaluate multiple models for symptom-based anemia prediction
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        output_dir: Directory to save results
        
    Returns:
        dict: Dictionary of model results
    """
    param_grids = {
        'Logistic Regression': {
            'C': [0.01, 0.1, 1, 10],
            'class_weight': [None, 'balanced'],
            'solver': ['liblinear', 'saga']
        },
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'class_weight': [None, 'balanced']
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'class_weight': [None, 'balanced']
        }
    }
    
    base_models = {
        'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=2000),
        'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE),
        'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
        'SVM': SVC(probability=True, random_state=RANDOM_STATE)
    }
    
    results = {}
    for name, model in base_models.items():
        print(f"\nTraining {name}...")
        
        # Hyperparameter tuning
        grid_search = GridSearchCV(
            model, 
            param_grids[name], 
            cv=5, 
            scoring='f1',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        results[name] = {
            'model': best_model,
            'accuracy': accuracy,
            'cv_accuracy': cv_scores.mean(),
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Anemic', 'Anemic'],
                   yticklabels=['Non-Anemic', 'Anemic'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix for {name}')
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'confusion_matrix_{name.replace(" ", "_")}.png'))
        else:
            plt.show()
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_prob'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Symptom-Based Anemia Prediction Models')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
    else:
        plt.show()
    
    return results

def analyze_feature_importance(models, X, output_dir=None):
    """
    Analyze feature importance for the Random Forest model
    
    Args:
        models: Dictionary of trained models
        X: Feature DataFrame
        output_dir: Directory to save visualizations
    """
    # Get Random Forest model
    rf_model = models['Random Forest']['model']
    
    # Calculate feature importances
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance for Anemia Prediction (Random Forest)')
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
    else:
        plt.show()
    
    return feature_importance

def compare_model_performance(results):
    """
    Compare performance of all models
    
    Args:
        results: Dictionary of model results
        
    Returns:
        DataFrame: Model comparison
    """
    # Create comparison DataFrame
    model_comparison = pd.DataFrame({
        'Model': list(results.keys()),
        'Test Accuracy': [results[model]['accuracy'] for model in results],
        'CV Accuracy': [results[model]['cv_accuracy'] for model in results],
    })
    
    # Sort by CV Accuracy
    model_comparison = model_comparison.sort_values('CV Accuracy', ascending=False).reset_index(drop=True)
    print("\nModel Comparison:")
    print(model_comparison)
    
    return model_comparison

def predict_anemia_risk(model, symptoms, scaler=None):
    """
    Predict anemia risk based on symptoms and risk factors,
    with special handling for the no-symptoms case
    
    Args:
        model: Trained model
        symptoms: Dictionary of symptoms and risk factors
        scaler: StandardScaler for preprocessing numeric features
        
    Returns:
        tuple: (prediction, probability)
    """
    # Strict check for no symptoms
    all_symptoms_zero = all(symptoms[key] == 0 for key in symptoms 
                          if key not in ['Gender', 'Age'])
    
    if all_symptoms_zero:
        return 0, 0.02  # Non-anemic with very low probability (2%)
    
    # Count how many symptoms the person has
    symptom_count = sum(symptoms[key] for key in symptoms if key not in ['Gender', 'Age'])
    
    # Convert input to DataFrame
    input_df = pd.DataFrame([symptoms])
    
    # Scale age if present and scaler is provided
    if 'Age' in input_df.columns and scaler is not None:
        input_df['Age'] = scaler.transform(input_df[['Age']])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    # Apply clinical knowledge adjustments
    key_anemia_symptoms = [
        'Pale_Skin', 'Fatigue', 'Shortness_of_Breath', 
        'Heavy_Periods', 'Recent_Blood_Loss'
    ]
    
    # If prediction is negative but patient has multiple key anemia symptoms, adjust probability
    if prediction == 0:
        key_symptom_count = sum(symptoms[k] for k in key_anemia_symptoms if k in symptoms)
        if key_symptom_count >= 3:
            probability = max(probability, 0.4)  # Increase probability to at least 40%
            if probability > 0.5:  # Potentially change prediction
                prediction = 1
    
    # If prediction is positive but symptom count is very low, reduce confidence
    if prediction == 1 and symptom_count <= 2:
        probability = min(probability, 0.7)  # Cap probability at 70%
    
    return prediction, probability

def create_pipeline_with_best_model(best_model):
    """
    Create a pipeline with preprocessing and the best model
    
    Args:
        best_model: Best performing model
        
    Returns:
        Pipeline: Scikit-learn pipeline
    """
    # Create a pipeline with scaling and the best model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', best_model)
    ])
    
    return pipeline

def save_model(model, model_path):
    """
    Save the model to disk
    
    Args:
        model: Model to save
        model_path: Path to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save the model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def run_symptom_model_training(output_dir=None, model_path=None):
    """
    Run the complete workflow for symptom-based anemia prediction
    
    Args:
        output_dir: Directory to save output
        model_path: Path to save the best model
    """
    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate synthetic dataset
    print("Generating synthetic symptom dataset...")
    symptom_df = generate_symptom_data(n_samples=2000)
    
    # Explore the data
    print("\nExploring symptom data...")
    feature_correlation = explore_symptom_data(symptom_df, output_dir)
    
    # Prepare data for modeling
    X = symptom_df.drop('Anemia_Status', axis=1)
    y = symptom_df['Anemia_Status']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    
    # Scale features (just Age)
    print("\nScaling numeric features...")
    non_binary_cols = ['Age']
    scaler = StandardScaler()
    X_train[non_binary_cols] = scaler.fit_transform(X_train[non_binary_cols])
    X_test[non_binary_cols] = scaler.transform(X_test[non_binary_cols])
    
    # Train and evaluate models
    print("\nTraining and evaluating models...")
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test, output_dir)
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    feature_importance = analyze_feature_importance(results, X, output_dir)
    
    # Compare model performance
    model_comparison = compare_model_performance(results)
    
    # Identify best model
    best_model_name = model_comparison.iloc[0]['Model']
    best_model = results[best_model_name]['model']
    
    print(f"\nBest model: {best_model_name}")
    
    # Create a pipeline with the best model
    if model_path:
        print("\nCreating pipeline with best model...")
        pipeline = create_pipeline_with_best_model(best_model)
        
        # Train on full dataset
        pipeline.fit(X, y)
        
        # Save the model
        save_model(pipeline, model_path)
    
    # Test with example cases
    print("\nTesting with example cases...")
    test_cases = [
        # Case 1: 45-year-old female with multiple anemia symptoms
        {
            'Gender': 1,  # Female
            'Age': 45,
            'Fatigue': 1, 
            'Shortness_of_Breath': 1,
            'Dizziness': 1,
            'Pale_Skin': 1,
            'Heart_Racing': 1,
            'Headaches': 1,
            'Cold_Hands_Feet': 1,
            'Brittle_Nails': 1,
            'Poor_Concentration': 1,
            'Heavy_Periods': 1,
            'Recent_Blood_Loss': 0,
            'Vegetarian_Diet': 0,
            'GI_Disorders': 0,
            'Chronic_Disease': 0
        },
        # Case 2: 35-year-old male with few symptoms
        {
            'Gender': 0,  # Male
            'Age': 35,
            'Fatigue': 0,
            'Shortness_of_Breath': 0,
            'Dizziness': 0,
            'Pale_Skin': 0,
            'Heart_Racing': 1,
            'Headaches': 1,
            'Cold_Hands_Feet': 0,
            'Brittle_Nails': 0,
            'Poor_Concentration': 0,
            'Heavy_Periods': 0,
            'Recent_Blood_Loss': 0,
            'Vegetarian_Diet': 1,
            'GI_Disorders': 0,
            'Chronic_Disease': 0
        },
        # Case 3: 65-year-old female with some concerning symptoms
        {
            'Gender': 1,  # Female
            'Age': 65,
            'Fatigue': 1,
            'Shortness_of_Breath': 1,
            'Dizziness': 0,
            'Pale_Skin': 0,
            'Heart_Racing': 0,
            'Headaches': 0,
            'Cold_Hands_Feet': 1,
            'Brittle_Nails': 0,
            'Poor_Concentration': 1,
            'Heavy_Periods': 0,
            'Recent_Blood_Loss': 0,
            'Vegetarian_Diet': 0,
            'GI_Disorders': 1,
            'Chronic_Disease': 1
        },
        # Case 4: No symptoms case
        {
            'Gender': 0,  # Male
            'Age': 40,
            'Fatigue': 0,
            'Shortness_of_Breath': 0,
            'Dizziness': 0,
            'Pale_Skin': 0,
            'Heart_Racing': 0,
            'Headaches': 0,
            'Cold_Hands_Feet': 0,
            'Brittle_Nails': 0,
            'Poor_Concentration': 0,
            'Heavy_Periods': 0,
            'Recent_Blood_Loss': 0,
            'Vegetarian_Diet': 0,
            'GI_Disorders': 0,
            'Chronic_Disease': 0
        }
    ]
    
    print(f"\nPredictions using {best_model_name}:\n")
    for i, case in enumerate(test_cases):
        prediction, probability = predict_anemia_risk(best_model, case, scaler)
        gender = "Female" if case['Gender'] == 1 else "Male"
        
        print(f"Case {i+1}: {gender}, {case['Age']} years old")
        print(f"Symptoms: {sum([v for k,v in case.items() if k not in ['Gender', 'Age']])} out of 14 symptoms/risk factors")
        print(f"Prediction: {'Anemic' if prediction == 1 else 'Non-Anemic'}")
        print(f"Probability of Anemia: {probability:.2%}\n")
    
    print("\nSymptom-Based Model Training and Evaluation Complete!")

if __name__ == "__main__":
    # Example usage
    output_dir = "./output/symptom_model_evaluation"
    model_path = "./models/symptom_anemia_prediction_model.pkl"
    
    run_symptom_model_training(output_dir, model_path) 