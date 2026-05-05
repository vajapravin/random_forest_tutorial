"""
Module for training the baseline Random Forest classifier.
"""

from typing import Optional
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import ipdb


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: int = 42
) -> RandomForestClassifier:
    """
    Train a Random Forest Classifier baseline model.
    
    Args:
        X_train: Preprocessed training features.
        y_train: Training labels.
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of the trees.
        random_state: Seed for reproducibility.
        
    Returns:
        Trained RandomForestClassifier model.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )

    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    from preprocess import load_dataset, split_data, preprocess_features
    
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled, _ = preprocess_features(X_train, X_test)
    
    model = train_random_forest(X_train_scaled, y_train)
    print("Baseline Random Forest model trained successfully!")
