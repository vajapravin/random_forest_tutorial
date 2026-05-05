"""
Module for model evaluation, classification metrics, confusion matrix, and feature importances.
"""

from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pandas as pd
import ipdb


def evaluate_model(
    model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, Any]:
    """
    Evaluate trained model performance on test set.
    
    Args:
        model: Trained model instance.
        X_test: Test features.
        y_test: True test labels.
        
    Returns:
        Dictionary containing evaluation metrics and confusion matrix.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_pred": y_pred
    }


def get_feature_importances(
    model: RandomForestClassifier, feature_names: pd.Index
) -> pd.DataFrame:
    """
    Extract and rank feature importances from the Random Forest model.
    
    Args:
        model: Trained Random Forest model.
        feature_names: Column names of the features.
        
    Returns:
        DataFrame sorted by feature importance descending.
    """
    importances = model.feature_importances_
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    return df_imp


if __name__ == "__main__":
    from preprocess import load_dataset, split_data, preprocess_features
    from train import train_random_forest
    
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, X_test_scaled, _ = preprocess_features(X_train, X_test)
    
    model = train_random_forest(X_train_scaled, y_train)
    results = evaluate_model(model, X_test_scaled, y_test)
    
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    
    print("\nTop 5 Important Features:")
    fi = get_feature_importances(model, X_train.columns)
    print(fi.head())
