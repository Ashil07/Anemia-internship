"""
preprocessing.py - Utility functions for data preprocessing

This file contains functions for preprocessing data before model training,
including feature engineering, scaling, and handling class imbalance.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.feature_config import encode_gender

# Explicitly expose SMOTE class from imblearn
from imblearn.over_sampling import SMOTE

def prepare_data_for_training(df, target_column='Decision_Class'):
    """
    Prepare data for model training by separating features and target
    
    Args:
        df: DataFrame containing the dataset
        target_column: Name of the target column
        
    Returns:
        X: Feature DataFrame
        y: Target Series
    """
    # Ensure gender is encoded
    if 'Gender_Encoded' not in df.columns and 'Gender' in df.columns:
        df['Gender_Encoded'] = df['Gender'].apply(encode_gender)
    
    # Define features and target
    X = df.drop(['Gender', target_column], axis=1)
    X['Gender_Encoded'] = df['Gender_Encoded']  # Add back gender as numeric
    y = df[target_column]
    
    return X, y

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

def scale_features(X_train, X_test=None, scaler=None):
    """
    Scale numerical features using StandardScaler
    
    Args:
        X_train: Training features
        X_test: Test features (optional)
        scaler: Pre-fit scaler (optional)
        
    Returns:
        X_train_scaled: Scaled training features
        X_test_scaled: Scaled test features (if provided)
        scaler: Fitted scaler
    """
    if scaler is None:
        scaler = StandardScaler()
    
    # Identify numerical columns (exclude encoded categorical features)
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    
    # Create copies to avoid modifying the originals
    X_train_scaled = X_train.copy()
    
    # Scale numerical features
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    
    if X_test is not None:
        X_test_scaled = X_test.copy()
        X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
        return X_train_scaled, X_test_scaled, scaler
    
    return X_train_scaled, scaler

def process_user_input(user_input):
    """
    Process user input for prediction
    
    Args:
        user_input: Dictionary containing user input
        
    Returns:
        processed_input: DataFrame with processed features
    """
    # Convert to dataframe if it's a dictionary
    if isinstance(user_input, dict):
        input_df = pd.DataFrame([user_input])
    else:
        input_df = user_input.copy()
    
    # Ensure Gender is encoded properly
    if 'Gender' in input_df.columns and 'Gender_Encoded' not in input_df.columns:
        input_df['Gender_Encoded'] = input_df['Gender'].apply(encode_gender)
        
    # Rename 'Gender' to 'Gender_Encoded' if needed
    if 'Gender' in input_df.columns and 'Gender_Encoded' not in input_df.columns:
        input_df['Gender_Encoded'] = input_df['Gender']
        input_df = input_df.drop('Gender', axis=1)
    
    return input_df