"""
Module for hyperparameter fine-tuning of Random Forest using GridSearchCV.
"""

from typing import Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd


def fine_tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Optional[Dict[str, Any]] = None,
    cv: int = 5,
    scoring: str = "accuracy"
) -> GridSearchCV:
    """
    Perform hyperparameter search using 5-fold cross-validation on the training set.
    
    Args:
        X_train: Training feature DataFrame.
        y_train: Target labels.
        param_grid: Dictionary of hyperparameters to search.
        cv: Number of cross-validation folds.
        scoring: Evaluation metric to optimize.
        
    Returns:
        Fitted GridSearchCV object containing the best model and parameters.
    """
    if param_grid is None:
        param_grid = {
            "n_estimators": [50, 100, 150],
            "max_depth": [None, 3, 5, 10],
            "min_samples_split": [2, 5],
            "criterion": ["gini", "entropy"]
        }
        
    base_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    return grid_search


if __name__ == "__main__":
    from preprocess import load_dataset, split_data, preprocess_features
    
    df = load_dataset()
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_scaled, _, _ = preprocess_features(X_train, X_test)
    
    print("Fine-tuning hyperparameters...")
    search = fine_tune_random_forest(X_train_scaled, y_train)
    print(f"Best Parameters: {search.best_params_}")
    print(f"Best CV Score: {search.best_score_:.4f}")
