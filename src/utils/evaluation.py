"""
evaluation.py - Utilities for model evaluation

This file contains functions for evaluating machine learning models,
including metrics calculation and visualization of results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
)
import os

def evaluate_model(model, X_test, y_test, model_name="Model", output_dir=None):
    """
    Evaluate a model and return performance metrics
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name: Name of the model for display
        output_dir: Directory to save plots (optional)
        
    Returns:
        results: Dictionary containing performance metrics
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Calculate ROC AUC if probability predictions are available
    roc_auc = None
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Print results
    print(f"\n--- {model_name} Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    if roc_auc:
        print(f"ROC AUC: {roc_auc:.4f}")
    
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(report)
    
    # Plot ROC curve if probability predictions are available
    if y_pred_proba is not None:
        plot_roc_curve(y_test, y_pred_proba, model_name, output_dir)
        plot_precision_recall_curve(y_test, y_pred_proba, model_name, output_dir)
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix,
        'classification_report': report,
        'model': model
    }

def plot_roc_curve(y_test, y_pred_proba, model_name="Model", output_dir=None):
    """
    Plot ROC curve for a model
    
    Args:
        y_test: Test target
        y_pred_proba: Predicted probabilities
        model_name: Name of the model
        output_dir: Directory to save the plot
    """
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc='lower right')
    
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, f'roc_curve_{model_name.replace(" ", "_").lower()}.png'), dpi=300)
    
    plt.show()

def plot_precision_recall_curve(y_test, y_pred_proba, model_name="Model", output_dir=None):
    """
    Plot Precision-Recall curve for a model
    
    Args:
        y_test: Test target
        y_pred_proba: Predicted probabilities
        model_name: Name of the model
        output_dir: Directory to save the plot
    """
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {model_name}')
    plt.legend(loc='lower left')
    
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, f'pr_curve_{model_name.replace(" ", "_").lower()}.png'), dpi=300)
    
    plt.show()

def plot_feature_importance(model, feature_names, model_name="Model", output_dir=None):
    """
    Plot feature importance for a model if available
    
    Args:
        model: Trained model
        feature_names: List of feature names
        model_name: Name of the model
        output_dir: Directory to save the plot
    """
    # For tree-based models
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title(f'Feature Importance from {model_name}')
        plt.tight_layout()
        
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.savefig(os.path.join(output_dir, f'feature_importance_{model_name.replace(" ", "_").lower()}.png'), dpi=300)
        
        plt.show()
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        return feature_importance
        
    # For linear models
    elif hasattr(model, 'coef_'):
        # For linear models
        coefficients = model.coef_[0]
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': np.abs(coefficients)
        }).sort_values('Coefficient', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Coefficient', y='Feature', data=feature_importance)
        plt.title(f'Feature Coefficients from {model_name}')
        plt.tight_layout()
        
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.savefig(os.path.join(output_dir, f'feature_coefficients_{model_name.replace(" ", "_").lower()}.png'), dpi=300)
        
        plt.show()
        
        print("\nFeature Coefficients:")
        print(feature_importance)
        
        return feature_importance
    
    return None

def compare_models(model_results, output_dir=None):
    """
    Compare multiple models based on evaluation metrics
    
    Args:
        model_results: Dictionary of model results keyed by model name
        output_dir: Directory to save the plot
    
    Returns:
        comparison_df: DataFrame with model comparison
    """
    comparison_data = []
    
    for name, results in model_results.items():
        comparison_data.append({
            'Model': name,
            'Accuracy': results['accuracy'],
            'ROC AUC': results.get('roc_auc', np.nan)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    sorted_df = comparison_df.sort_values('ROC AUC', ascending=False)
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    # Plot accuracy
    plt.subplot(2, 1, 1)
    sns.barplot(x='Model', y='Accuracy', data=sorted_df)
    plt.title('Model Comparison - Accuracy')
    plt.ylim([0.7, 1.0])  # Adjustable based on your results
    plt.xticks(rotation=45)
    
    # Plot ROC AUC
    plt.subplot(2, 1, 2)
    sns.barplot(x='Model', y='ROC AUC', data=sorted_df)
    plt.title('Model Comparison - ROC AUC')
    plt.ylim([0.7, 1.0])  # Adjustable based on your results
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)
    
    plt.show()
    
    print("\nModel Comparison:")
    print(sorted_df)
    
    return sorted_df

def plot_confusion_matrix(y_test, y_pred, model_name="Model", output_dir=None):
    """
    Plot confusion matrix for model predictions
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        output_dir: Directory to save the plot
    """
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Non-Anemic", "Anemic"],
                yticklabels=["Non-Anemic", "Anemic"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name}")
    
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png'), dpi=300)
    
    plt.show()