"""
model_training.py - Train and evaluate machine learning models for anemia prediction

This script trains multiple machine learning models, evaluates their performance,
and saves the best model for later use in the web application.
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
from config.feature_config import encode_gender
from src.utils.preprocessing import prepare_data_for_training
# Directly import SMOTE to fix the error
from imblearn.over_sampling import SMOTE
from src.utils.evaluation import (
    evaluate_model, plot_feature_importance, compare_models
)

# Machine learning libraries
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from imblearn.pipeline import Pipeline as ImbPipeline

# Set random state for reproducibility
RANDOM_STATE = 42

def load_and_prepare_data(data_path):
    """
    Load and prepare data for model training
    
    Args:
        data_path: Path to the anemia dataset
        
    Returns:
        X: Feature DataFrame
        y: Target Series
        X_train, X_test, y_train, y_test: Train and test splits
    """
    try:
        print("Loading and preparing the dataset...")
        df = pd.read_excel(data_path)
        
        # Convert gender to binary
        gender_mapping = {'f': 1, 'm': 0}
        df['Gender_Encoded'] = df['Gender'].map(gender_mapping)
        
        # Define X and y
        X = df.drop(['Gender', 'Decision_Class'], axis=1)
        X['Gender_Encoded'] = df['Gender_Encoded']  # Add back gender as numeric
        y = df['Decision_Class']
        
        # Check class distribution
        print(f"Class distribution: {pd.Series(y).value_counts(normalize=True).round(3) * 100}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
        )
        print(f"Training set shape: {X_train.shape}")
        print(f"Testing set shape: {X_test.shape}")
        
        return X, y, X_train, X_test, y_train, y_test
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None, None

def apply_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE to handle class imbalance in the training set
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random seed for reproducibility
        
    Returns:
        X_resampled: Resampled features
        y_resampled: Resampled target
    """
    print("Applying SMOTE to handle class imbalance...")
    print(f"Original class distribution: {pd.Series(y_train).value_counts(normalize=True).round(3) * 100}")
    
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Resampled class distribution: {pd.Series(y_resampled).value_counts(normalize=True).round(3) * 100}")
    
    return X_resampled, y_resampled

def perform_feature_selection(X_train, y_train, output_dir=None):
    """
    Perform feature selection using multiple methods
    
    Args:
        X_train: Training features
        y_train: Training target
        output_dir: Directory to save plots
        
    Returns:
        feature_importances: Dictionary of feature importance results
    """
    print("\n1. Feature Selection Analysis:")
    
    # 1. Univariate Selection - ANOVA F-value
    print("\n1.1 Univariate Selection with ANOVA F-value:")
    selector_f = SelectKBest(f_classif, k='all')
    selector_f.fit(X_train, y_train)
    f_scores = pd.DataFrame({
        'Feature': X_train.columns,
        'F_Score': selector_f.scores_,
        'P_Value': selector_f.pvalues_
    })
    print(f_scores.sort_values('F_Score', ascending=False))
    
    # 2. Mutual Information
    print("\n1.2 Mutual Information Selection:")
    selector_mi = SelectKBest(mutual_info_classif, k='all')
    selector_mi.fit(X_train, y_train)
    mi_scores = pd.DataFrame({
        'Feature': X_train.columns,
        'MI_Score': selector_mi.scores_
    })
    print(mi_scores.sort_values('MI_Score', ascending=False))
    
    # 3. Feature Importance with Random Forest
    print("\n1.3 Feature Importance with Random Forest:")
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    rf_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': rf.feature_importances_
    })
    print(rf_importance.sort_values('Importance', ascending=False))
    
    # 4. Recursive Feature Elimination
    print("\n1.4 Recursive Feature Elimination with Cross-Validation:")
    estimator = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    rfecv = RFECV(estimator=estimator, step=1, cv=StratifiedKFold(5), scoring='accuracy')
    rfecv.fit(X_train, y_train)
    print(f"Optimal number of features: {rfecv.n_features_}")
    print(f"Features selected: {X_train.columns[rfecv.support_].tolist()}")
    
    # Plot feature importance from different methods
    plt.figure(figsize=(14, 10))
    
    # ANOVA F-scores
    plt.subplot(3, 1, 1)
    sns.barplot(x='Feature', y='F_Score', data=f_scores.sort_values('F_Score', ascending=False))
    plt.title('Feature Importance: ANOVA F-values')
    plt.xticks(rotation=45)
    
    # Mutual Information
    plt.subplot(3, 1, 2)
    sns.barplot(x='Feature', y='MI_Score', data=mi_scores.sort_values('MI_Score', ascending=False))
    plt.title('Feature Importance: Mutual Information')
    plt.xticks(rotation=45)
    
    # Random Forest Importance
    plt.subplot(3, 1, 3)
    sns.barplot(x='Feature', y='Importance', data=rf_importance.sort_values('Importance', ascending=False))
    plt.title('Feature Importance: Random Forest')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
    plt.show()
    
    return {
        'f_scores': f_scores,
        'mi_scores': mi_scores,
        'rf_importance': rf_importance,
        'rfecv': rfecv
    }

def train_and_evaluate_models(X_train, y_train, X_test, y_test, output_dir=None):
    """
    Train and evaluate multiple machine learning models
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        output_dir: Directory to save results
        
    Returns:
        model_results: Dictionary of model evaluation results
    """
    print("\n2. Model Training and Evaluation:")
    
    # Define models to evaluate
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'SVM': SVC(probability=True, random_state=RANDOM_STATE),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=RANDOM_STATE)
    }
    
    # Use SMOTE to handle class imbalance
    print("\nApplying SMOTE to handle class imbalance in the training set...")
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train, RANDOM_STATE)
    
    # Train and evaluate each model
    model_results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        # Train on resampled data
        model.fit(X_train_resampled, y_train_resampled)
        
        # Manually calculate accuracy and ROC AUC (as a backup)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate ROC AUC if possible
        try:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            else:
                # Use decision function if available, otherwise use predictions
                if hasattr(model, 'decision_function'):
                    y_score = model.decision_function(X_test)
                else:
                    y_score = y_pred
                roc_auc = roc_auc_score(y_test, y_score)
        except Exception as e:
            print(f"Warning: Could not calculate ROC AUC for {name}: {e}")
            roc_auc = None
        
        # Store these values for fallback
        basic_metrics = {
            'model': model,
            'accuracy': accuracy,
            'roc_auc': roc_auc
        }
        
        # Evaluate on test set with our helper function
        try:
            results = evaluate_model(model, X_test, y_test, model_name=name, output_dir=output_dir)
            # Merge with basic metrics in case the helper function had issues
            results.update({k: v for k, v in basic_metrics.items() if k not in results})
            model_results[name] = results
        except Exception as e:
            print(f"Warning: Error during model evaluation: {e}")
            model_results[name] = basic_metrics
    
    # Compare model performances
    print("\n3. Model Comparison:")
    compare_models(model_results, output_dir)
    
    return model_results

def select_best_model(model_results, metric='accuracy'):
    """
    Select the best model based on the specified metric
    
    Args:
        model_results: Dictionary of model evaluation results
        metric: Metric to use for selection (default: 'accuracy')
        
    Returns:
        best_model_name: Name of the best model
        best_model: The best model object
    """
    # Create comparison DataFrame
    comparison_data = []
    
    for name, results in model_results.items():
        model_data = {
            'Model': name,
            'Accuracy': results.get('accuracy', 0)
        }
        
        # Add ROC AUC if available
        if 'roc_auc' in results and results['roc_auc'] is not None:
            model_data['ROC AUC'] = results['roc_auc']
        
        comparison_data.append(model_data)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Check if the requested metric is available
    if metric not in comparison_df.columns:
        print(f"Warning: Metric '{metric}' not available. Using 'Accuracy' instead.")
        metric = 'Accuracy'
    
    # Sort by the specified metric
    sorted_df = comparison_df.sort_values(metric, ascending=False)
    print("\nModel comparison based on", metric)
    print(sorted_df)
    
    # Select best model
    best_model_name = sorted_df.iloc[0]['Model']
    best_model = model_results[best_model_name]['model']
    
    print(f"\nBest model: {best_model_name}")
    
    return best_model_name, best_model

def create_pipeline_with_best_model(best_model, random_state=42):
    """
    Create a pipeline with SMOTE and the best model
    
    Args:
        best_model: Best model object
        random_state: Random seed for reproducibility
        
    Returns:
        pipeline: Pipeline with SMOTE and the best model
    """
    # Create a pipeline with SMOTE and the best model
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=random_state)),
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

def run_model_training(data_path, output_dir, model_path):
    """
    Run the complete model training workflow
    
    Args:
        data_path: Path to the dataset
        output_dir: Directory to save results
        model_path: Path to save the best model
    """
    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load and prepare data
    X, y, X_train, X_test, y_train, y_test = load_and_prepare_data(data_path)
    if X is None:
        return
    
    # Perform feature selection
    feature_importances = perform_feature_selection(X_train, y_train, output_dir)
    
    # Train and evaluate models
    model_results = train_and_evaluate_models(X_train, y_train, X_test, y_test, output_dir)
    
    # Select the best model
    best_model_name, best_model = select_best_model(model_results)
    
    # Create pipeline with the best model
    pipeline = create_pipeline_with_best_model(best_model)
    
    # Retrain the pipeline on all data
    pipeline.fit(X, y)
    
    # Save the model
    save_model(pipeline, model_path)
    
    # Display feature importance for the best model if applicable
    if hasattr(best_model, 'feature_importances_') or hasattr(best_model, 'coef_'):
        print("\nFeature Importance from Best Model:")
        feature_importance = plot_feature_importance(best_model, X.columns, model_name=best_model_name, output_dir=output_dir)
    
    print("\nModel Training and Evaluation Complete!")

if __name__ == "__main__":
    # Example usage
    data_path = "./data/Anemia Dataset.xlsx"
    output_dir = "./output/model_evaluation"
    model_path = "./models/anemia_prediction_model.pkl"
    
    run_model_training(data_path, output_dir, model_path)