import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from src.utils import load_data

def preprocess_data(df, use_smote=True):
    """
    Preprocess the telecom dataset using only specified columns: CustomerID, Age, Gender, Tenure, 
    MonthlyCharges, ContractType, InternetService, TechSupport, TotalCharges, Churn.
    Handle missing values, encode categorical variables, perform feature engineering, 
    scale numerical features, and optionally apply SMOTE.
    """
    df = df.copy()
    
    # Select and rename columns to match user request
    # Adjusted to handle if Age is already present instead of SeniorCitizen
    columns_to_keep = {
        'customerID': 'CustomerID',
        'SeniorCitizen': 'Age',  # Will fall back if SeniorCitizen exists, otherwise try Age
        'gender': 'Gender',
        'tenure': 'Tenure',
        'MonthlyCharges': 'MonthlyCharges',
        'Contract': 'ContractType',
        'InternetService': 'InternetService',
        'TechSupport': 'TechSupport',
        'TotalCharges': 'TotalCharges',
        'Churn': 'Churn'
    }
    available_cols = [col for col in columns_to_keep.keys() if col in df.columns]
    if 'SeniorCitizen' not in df.columns and 'Age' in df.columns:
        columns_to_keep['Age'] = 'Age'  # Use Age if SeniorCitizen is missing
        available_cols = [col if col != 'SeniorCitizen' else 'Age' for col in available_cols]
    df = df[available_cols].rename(columns={k: v for k, v in columns_to_keep.items() if k in available_cols})

    # Handle missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['Gender', 'ContractType', 'InternetService', 'TechSupport', 'Churn']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            print(f"Warning: Column '{col}' not found in dataset, skipping encoding.")

    # Feature engineering: TenureGroup from Tenure
    df['Tenure'] = pd.to_numeric(df['Tenure'], errors='coerce').fillna(0).clip(lower=0)
    df['TenureGroup'] = pd.cut(df['Tenure'], bins=[0, 12, 24, 36, 48, 60, 72, np.inf], 
                              labels=range(1, 8), right=True)
    df['TenureGroup'] = df['TenureGroup'].cat.codes + 1
    df['TenureGroup'] = df['TenureGroup'].fillna(1).astype(int)

    # Drop CustomerID (not for training)
    if 'CustomerID' in df.columns:
        df.drop('CustomerID', axis=1, inplace=True)

    # Split features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['Tenure', 'MonthlyCharges', 'TotalCharges']
    available_numerical = [col for col in numerical_cols if col in X.columns]
    if available_numerical:
        X[available_numerical] = scaler.fit_transform(X[available_numerical])
        print(f"Scaled numerical columns: {available_numerical}")
    else:
        print("Warning: No numerical columns found for scaling.")

    # Apply SMOTE if requested
    if use_smote:
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
        print(f"Applied SMOTE: New dataset shape: {X.shape}, Class distribution: {pd.Series(y).value_counts().to_dict()}")

    return X, y, scaler, label_encoders, available_numerical

if __name__ == "__main__":
    df = load_data()
    X, y, scaler, label_encoders, numerical_cols = preprocess_data(df)
    print("Preprocessing complete. Sample features:\n", X.head())
    print(f"Numerical columns scaled: {numerical_cols}")
    print(f"TenureGroup sample (first 5): {X['TenureGroup'].head().tolist()}")
    print(f"Any NaNs in TenureGroup? {X['TenureGroup'].isna().any()}")
    print(f"Max Tenure: {X['Tenure'].max()}")  # Updated to use X instead of df