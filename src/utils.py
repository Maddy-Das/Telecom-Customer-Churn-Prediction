import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import os

def load_data(file_path='/home/maddy-das/myproject/filtered_dataset.csv'):
    """
    Load dataset from CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please ensure customer_data.csv is in the data/ directory.")

def evaluate_model(y_test, y_pred):
    """
    Evaluate model performance using multiple metrics.
    """
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_pred)
    }
    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm

def save_model(model, file_name):
    """
    Save model to models/ directory.
    """
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f'models/{file_name}')

def load_model(file_name):
    """
    Load model from models/ directory.
    """
    try:
        return joblib.load(f'models/{file_name}')
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file {file_name} not found in models/ directory.")