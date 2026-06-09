"""
preprocess.py
Data cleaning, encoding, and class imbalance handling
for Credit Card Churn Prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV dataset."""
    df = pd.read_csv(filepath)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that don't contribute to prediction."""
    cols_to_drop = [
        "CLIENTNUM",
        "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
        "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
    ]
    existing = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing)
    print(f"Dropped {len(existing)} irrelevant columns.")
    return df


def encode_target(df: pd.DataFrame, target_col: str = "Attrition_Flag") -> pd.DataFrame:
    """Encode target: Attrited Customer=1, Existing Customer=0."""
    df[target_col] = df[target_col].map(
        {"Attrited Customer": 1, "Existing Customer": 0}
    )
    print(f"Target distribution:\n{df[target_col].value_counts()}")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label encode binary columns, one-hot encode multi-class columns."""
    # Binary categorical columns
    binary_cols = ["Gender"]
    le = LabelEncoder()
    for col in binary_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    # Multi-class categorical columns
    multi_cols = ["Education_Level", "Marital_Status", "Income_Category", "Card_Category"]
    df = pd.get_dummies(df, columns=[c for c in multi_cols if c in df.columns], drop_first=True)

    print(f"Encoded categoricals. Shape: {df.shape}")
    return df


def split_data(df: pd.DataFrame, target_col: str = "Attrition_Flag", test_size: float = 0.25, random_state: int = 42):
    """Split into train/test sets."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train, random_state: int = 42):
    """Oversample minority class using SMOTE."""
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE — X: {X_res.shape}, y distribution: {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res


def apply_undersampling(X_train, y_train, random_state: int = 42):
    """Undersample majority class."""
    rus = RandomUnderSampler(random_state=random_state)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    print(f"After undersampling — X: {X_res.shape}, y distribution: {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res


def preprocess_pipeline(filepath: str, sampling: str = "smote"):
    """
    Full preprocessing pipeline.
    
    Args:
        filepath: Path to raw CSV file
        sampling: 'smote', 'undersample', or 'none'
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    df = load_data(filepath)
    df = drop_irrelevant_columns(df)
    df = encode_target(df)
    df = encode_categoricals(df)
    X_train, X_test, y_train, y_test = split_data(df)

    if sampling == "smote":
        X_train, y_train = apply_smote(X_train, y_train)
    elif sampling == "undersample":
        X_train, y_train = apply_undersampling(X_train, y_train)

    return X_train, X_test, y_train, y_test
