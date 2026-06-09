"""
train.py
Model training pipeline for Credit Card Churn Prediction.
Trains and compares multiple classifiers, then saves the best model.
"""

import os
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, roc_auc_score, classification_report

from preprocess import preprocess_pipeline


DATA_PATH = "../data/BankChurners.csv"
MODEL_SAVE_PATH = "../models/best_model.pkl"


def get_models() -> dict:
    """Return dictionary of models to compare."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
    }


def compare_models(models: dict, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """Train all models and compare recall & AUC on test set."""
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        results.append({"Model": name, "Recall": round(recall, 4), "AUC": round(auc, 4)})
        print(f"{name:25s} | Recall: {recall:.4f} | AUC: {auc:.4f}")

    results_df = pd.DataFrame(results).sort_values("Recall", ascending=False)
    return results_df


def tune_xgboost(X_train, y_train) -> XGBClassifier:
    """Grid search hyperparameter tuning for XGBoost."""
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    }
    xgb = XGBClassifier(eval_metric="logloss", random_state=42)
    grid_search = GridSearchCV(xgb, param_grid, cv=3, scoring="recall", n_jobs=1, verbose=1)
    grid_search.fit(X_train, y_train)
    print(f"\nBest XGBoost params: {grid_search.best_params_}")
    return grid_search.best_estimator_


def save_model(model, path: str):
    """Save trained model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")


def main():
    print("=" * 50)
    print("Credit Card Churn Prediction — Training Pipeline")
    print("=" * 50)

    # Preprocessing
    X_train, X_test, y_train, y_test = preprocess_pipeline(DATA_PATH, sampling="smote")

    # Compare models
    print("\n--- Model Comparison ---")
    models = get_models()
    results_df = compare_models(models, X_train, X_test, y_train, y_test)
    print(f"\nResults:\n{results_df.to_string(index=False)}")

    # Tune best model (XGBoost)
    print("\n--- Tuning XGBoost ---")
    best_model = tune_xgboost(X_train, y_train)

    # Final evaluation
    y_pred = best_model.predict(X_test)
    print("\n--- Final Model Evaluation ---")
    print(classification_report(y_test, y_pred, target_names=["Existing", "Churned"]))

    # Save model
    save_model(best_model, MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()
