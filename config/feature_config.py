"""
feature_config.py - Centralized configuration for anemia prediction project

This file defines feature names, transformations, and normal ranges
to ensure consistency between training and inference.
"""

# Feature definitions
FEATURES = {
    'Age': {
        'type': 'numerical',
        'min_value': 1,
        'max_value': 100,
        'default': 35
    },
    'Gender': {
        'type': 'categorical',
        'mapping': {'f': 1, 'm': 0, 'Female': 1, 'Male': 0},
        'encoded_name': 'Gender_Encoded'
    },
    'Hb': {
        'type': 'numerical',
        'min_value': 5.0,
        'max_value': 20.0,
        'default': 12.0,
        'unit': 'g/dL'
    },
    'RBC': {
        'type': 'numerical',
        'min_value': 1.0,
        'max_value': 8.0,
        'default': 4.5,
        'unit': 'million/μL'
    },
    'PCV': {
        'type': 'numerical',
        'min_value': 10.0,
        'max_value': 60.0,
        'default': 35.0,
        'unit': '%'
    },
    'MCV': {
        'type': 'numerical',
        'min_value': 20.0,
        'max_value': 120.0,
        'default': 85.0,
        'unit': 'fL'
    },
    'MCH': {
        'type': 'numerical',
        'min_value': 10.0,
        'max_value': 40.0,
        'default': 28.0,
        'unit': 'pg'
    },
    'MCHC': {
        'type': 'numerical',
        'min_value': 25.0,
        'max_value': 40.0,
        'default': 33.0,
        'unit': 'g/dL'
    }
}

# Target variable
TARGET = 'Decision_Class'

# Normal ranges based on medical literature
NORMAL_RANGES = {
    'Hb': {'f': [12.0, 15.5], 'm': [13.5, 17.5]},
    'RBC': {'f': [4.0, 5.2], 'm': [4.5, 5.9]},
    'PCV': {'f': [37.0, 47.0], 'm': [40.0, 52.0]},
    'MCV': {'all': [80.0, 100.0]},
    'MCH': {'all': [27.0, 33.0]},
    'MCHC': {'all': [32.0, 36.0]}
}

# Feature processing functions
def encode_gender(gender_value):
    """Convert gender to encoded value"""
    gender_mapping = FEATURES['Gender']['mapping']
    return gender_mapping.get(gender_value, None)

def prepare_features_for_model(input_data):
    """Process input data to prepare for model prediction"""
    if isinstance(input_data, dict):
        # Handle single input as dictionary
        result = {}
        
        # Process gender if present
        if 'Gender' in input_data:
            result['Gender_Encoded'] = encode_gender(input_data['Gender'])
        elif 'gender' in input_data:
            result['Gender_Encoded'] = encode_gender(input_data['gender'])
            
        # Copy other features
        for feature, config in FEATURES.items():
            if feature != 'Gender' and feature in input_data:
                result[feature] = input_data[feature]
                
        return result
    else:
        # Handle DataFrame input
        # Implementation depends on your specific needs
        return input_data

def get_feature_names_for_model():
    """Return list of feature names expected by the model"""
    names = []
    for feature, config in FEATURES.items():
        if feature == 'Gender':
            names.append(config['encoded_name'])
        else:
            names.append(feature)
    return names