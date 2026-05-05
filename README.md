# 🌲 Machine Learning with Random Forest: End-to-End Tutorial

Welcome to this beginner-friendly, industry-standard project designed to teach you the fundamentals of Machine Learning using **Random Forest**.

This sample project covers the full lifecycle of a Machine Learning model:
1. **Data Loading & Exploration**
2. **Dataset Preprocessing & Scaling** (Preventing Data Leakage)
3. **Model Training** (Baseline Random Forest)
4. **Hyperparameter Fine-Tuning** (Cross-Validation with `GridSearchCV`)
5. **Evaluation & Metrics** (Accuracy, Confusion Matrix, Classification Report)
6. **Feature Importance Analysis**

---

## 📂 Project Structure

```text
random_forest_tutorial/
├── preprocess.py   # Step 1 & 2: Dataset loading, splitting, and scaling
├── train.py        # Step 3: Training the baseline Random Forest classifier
├── tune.py         # Step 4: Fine-tuning hyperparameters with GridSearchCV
├── evaluate.py     # Step 5 & 6: Evaluating performance metrics and feature importance
├── main.py         # Main execution pipeline (runs the entire workflow end-to-end)
└── README.md       # Step-by-step learning guide
```

---

## 🚀 Getting Started

### 1. Prerequisites & Dependencies

Make sure Python 3.8+ is installed. Install required packages using `pip`:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

### 2. Running the Full Pipeline

To execute the entire project end-to-end, simply run:

```bash
python main.py
```

---

## 🎓 Step-by-Step Learning Guide

### Step 1: Loading the Dataset (`preprocess.py`)
We use the **Wine Recognition Dataset** from `scikit-learn`. It consists of 178 samples of wine with 13 continuous chemical attributes (like alcohol content, malic acid, flavonoids) categorized into 3 class labels.

### Step 2: Data Splitting & Preprocessing (`preprocess.py`)
* **Data Leakage Prevention**: We **MUST** split the dataset into `Train` (80%) and `Test` (20%) sets *before* applying any preprocessing transformers.
* **Standard Scaling**: Numerical features are standardized ($\mu=0, \sigma=1$). The `StandardScaler` is fitted **strictly on the training data**, and then used to transform both train and test sets.

### Step 3: Model Training (`train.py`)
* **What is Random Forest?**: An ensemble method that constructs multiple Decision Trees during training and outputs the majority class vote.
* **Baseline Training**: We instantiate `RandomForestClassifier(n_estimators=100)` and train it on `X_train`.

### Step 4: Model Fine-Tuning (`tune.py`)
* **What is Fine-Tuning?**: Finding optimal hyperparameters to maximize model accuracy.
* **GridSearchCV**: We perform a 5-fold cross-validation grid search over:
  * `n_estimators`: Number of decision trees (e.g. 50, 100, 150).
  * `max_depth`: Maximum depth of trees (e.g. None, 3, 5, 10).
  * `criterion`: Splitting quality metric (`gini` vs `entropy`).

### Step 5: Evaluation (`evaluate.py`)
* **Accuracy Score**: Proportion of correct predictions.
* **Confusion Matrix**: A table visualising actual vs predicted labels across all classes.
* **Feature Importances**: Ranks which chemical attributes contributed most to classifying the wines (e.g. `flavanoids`, `color_intensity`, `proline`).

---

## 🏆 Key Concepts Summary

| Concept | Description |
|---|---|
| **Data Leakage** | Occurs when test set information leaks into model training (e.g., scaling before splitting). Avoided by fitting transformers only on training data. |
| **Stratified Split** | Ensures train and test splits retain the same class distributions as the original dataset. |
| **GridSearchCV** | Automatically searches through hyperparameter combinations using Cross-Validation on training data. |
| **Feature Importance** | Measures how much each feature reduces tree impurity across the entire ensemble. |

---

## 🛠️ Industry Best Practices Followed
* **PEP 8 Compliance**: Clean code style, type hinting (`typing`), and explicit function docstrings.
* **Modular Code Architecture**: Separated concerns into modular files (`preprocess`, `train`, `tune`, `evaluate`).
* **Reproducibility**: Set explicit `random_state` seeds across splits and estimators.
