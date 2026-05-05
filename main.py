"""
Main pipeline execution entry point. Runs complete workflow:
1. Load dataset
2. Split dataset into train/test
3. Preprocess features (standard scaling)
4. Train baseline Random Forest
5. Fine-tune hyperparameters via Cross-Validation
6. Evaluate both baseline and tuned models on test set
7. Output summary metrics and feature importances
"""

from preprocess import load_dataset, split_data, preprocess_features
from train import train_random_forest
from tune import fine_tune_random_forest
from evaluate import evaluate_model, get_feature_importances
import ipdb

def main():
    print("=" * 60)
    print("      RANDOM FOREST END-TO-END PIPELINE TUTORIAL      ")
    print("=" * 60)

    # 1. Load Data
    print("\n[Step 1] Loading Dataset...")
    df = load_dataset()
    print(f"Loaded Wine Recognition dataset with shape: {df.shape}")
    print(f"Target distribution:\n{df['wine_class'].value_counts().to_dict()}")

    # 2. Split Data
    print("\n[Step 2] Splitting Data into Train & Test sets (80/20)...")
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)
    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 3. Preprocess Features
    print("\n[Step 3] Preprocessing Features (StandardScaling)...")
    X_train_scaled, X_test_scaled, scaler = preprocess_features(X_train, X_test)
    print("Scaler fitted strictly on training data and applied to both sets.")

    # 4. Train Baseline Model
    print("\n[Step 4] Training Baseline Random Forest Model...")
    baseline_model = train_random_forest(X_train_scaled, y_train, random_state=42)
    print("Baseline model trained.")

    # 5. Evaluate Baseline Model
    baseline_results = evaluate_model(baseline_model, X_test_scaled, y_test)
    print(f"-> Baseline Test Accuracy: {baseline_results['accuracy'] * 100:.2f}%")

    # 6. Hyperparameter Fine-Tuning
    print("\n[Step 5] Fine-Tuning Model Hyperparameters with 5-Fold Cross-Validation...")
    grid_search = fine_tune_random_forest(X_train_scaled, y_train, cv=5)
    best_model = grid_search.best_estimator_
    print(f"Best Parameters Found: {grid_search.best_params_}")
    print(f"Best CV Training Score: {grid_search.best_score_ * 100:.2f}%")

    # 7. Evaluate Fine-Tuned Model
    print("\n[Step 6] Evaluating Fine-Tuned Model on Unseen Test Data...")
    tuned_results = evaluate_model(best_model, X_test_scaled, y_test)
    print(f"-> Fine-Tuned Test Accuracy: {tuned_results['accuracy'] * 100:.2f}%")
    print("\nConfusion Matrix (Fine-Tuned Model):")
    print(tuned_results["confusion_matrix"])

    # 8. Feature Importance
    print("\n[Step 7] Feature Importance Analysis:")
    fi_df = get_feature_importances(best_model, X_train.columns)
    for idx, row in fi_df.head(5).iterrows():
        print(f"  {idx+1}. {row['Feature']:<25}: {row['Importance']:.4f}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
