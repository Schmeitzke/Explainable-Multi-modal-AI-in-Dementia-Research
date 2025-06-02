import pickle
import json
import pandas as pd

def save_object(obj, filepath):
    """Saves a Python object to a file using pickle."""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Object saved to {filepath}")

def load_object(filepath):
    """Loads a Python object from a pickle file."""
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    print(f"Object loaded from {filepath}")
    return obj

def save_encoded_columns(columns, filepath):
    """Saves a list of one-hot encoded column names to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(columns, f)
    print(f"Encoded column names saved to {filepath}")

def load_encoded_columns(filepath):
    """Loads a list of one-hot encoded column names from a JSON file."""
    with open(filepath, 'r') as f:
        columns = json.load(f)
    print(f"Encoded column names loaded from {filepath}")
    return columns

def verify_target_mapping(df: pd.DataFrame, target_column: str):
    """Verifies that the target column contains 0, 1, and 2."""
    expected_values = [0, 1, 2]
    
    if df[target_column].isnull().any():
        print(f"Warning: Target column '{target_column}' contains NaN values. These should be handled or filtered.")
        
    actual_unique_values = sorted([val for val in df[target_column].unique() if pd.notna(val)])

    if actual_unique_values == expected_values:
        print(f"Target column '{target_column}' mapping is correct (0, 1, 2).")
    else:
        print(f"Warning: Target column '{target_column}' mapping is INCORRECT or has unexpected values.")
        print(f"Expected unique values: {expected_values}")
        print(f"Actual unique values in data: {actual_unique_values}")
        raise ValueError(f"Target column '{target_column}' has incorrect mapping. Expected {expected_values}, got {actual_unique_values}")