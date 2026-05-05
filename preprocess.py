"""
Module for loading and preprocessing the dataset for Random Forest training.
Follows PEP 8 guidelines and strict Machine Learning practices (preventing data leakage).
"""

from typing import Tuple
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import ipdb


def load_dataset() -> pd.DataFrame:
    """
    Load the Wine Recognition dataset from scikit-learn.
    
    Returns:
        pd.DataFrame: DataFrame containing features and target column.
    """
    raw_data = load_wine(as_frame=True)
    df = raw_data.frame
    # Rename target column for clarity
    df = df.rename(columns={"target": "wine_class"})
    return df


def split_data(
    df: pd.DataFrame, target_column: str = "wine_class", test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing feature/target sets.
    
    Args:
        df: Input DataFrame.
        target_column: Name of the target label column.
        test_size: Proportion of the dataset to include in the test split.
        random_state: Seed for reproducibility.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Stratified split to maintain class balance in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def preprocess_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Preprocess features by fitting a StandardScaler strictly on the training set
    and transforming both training and test sets to prevent data leakage.
    
    Args:
        X_train: Training features.
        X_test: Testing features.
        
    Returns:
        Tuple of (X_train_scaled_df, X_test_scaled_df, scaler_object)
    """
    scaler = StandardScaler()

    # Fit strictly on training data
    X_train_scaled = scaler.fit_transform(X_train)
    # Transform test data using the fitted scaler
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for readability and column preservation
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    return X_train_scaled_df, X_test_scaled_df, scaler


if __name__ == "__main__":
    print("Testing Preprocessing Pipeline...")
    df = load_dataset()
    print(f"Dataset Loaded. Shape: {df.shape}")
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled, scaler = preprocess_features(X_train, X_test)
    print(f"X_train shape: {X_train_scaled.shape}, X_test shape: {X_test_scaled.shape}")
    print("Preprocessing completed successfully!")
