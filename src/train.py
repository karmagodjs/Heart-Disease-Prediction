import numpy as np
import os
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.pipeline import Pipeline
from src.data_preprocessing import load_data, prepare_data


os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)
df = load_data()
(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
    numerical,
    categorical
) = prepare_data(df)

models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        {
            "classifier__C": [0.1, 1, 10]
        }
    ),
    "Random Forest": (
        RandomForestClassifier(
            random_state=42,
            class_weight="balanced"
        ),
        {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 5, 10]
        }
    ),
    "XGBoost": (
        XGBClassifier(
            eval_metric="logloss",
            random_state=42
        ),
        {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [3, 5],
            "classifier__learning_rate": [0.05, 0.1]
        }
    )
}
results = []
trained_models = {}

for name, (classifier, params) in models.items():
    print(f"\nTraining {name}...")
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    grid = GridSearchCV(
        pipeline,
        params,
        cv=5,
        scoring="recall",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)[:, 1]
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC AUC": roc_auc_score(y_test, probabilities),
        "Best CV Recall": grid.best_score_
    }
    results.append(metrics)
    trained_models[name] = best_model
    print("Best parameters:", grid.best_params_)
    print("Test recall:", metrics["Recall"])

comparison = pd.DataFrame(results)
comparison.to_csv("reports/model_comparison.csv", index=False)
print("\n========== MODEL COMPARISON ==========")
print(comparison.to_string(index=False))

best_name = comparison.loc[
    comparison["Best CV Recall"].idxmax(), "Model"
]
best_model = trained_models[best_name]
best_model.features = X_train.columns.tolist()
best_model.numerical = numerical
best_model.categorical = categorical
best_model.feature_values = {
    col: sorted(X_train[col].dropna().unique().tolist(), key=str)
    for col in categorical
}
best_model.defaults = {
    col: float(X_train[col].median())
    for col in numerical
}
best_model.categorical_defaults = {
    col: X_train[col].mode().iloc[0]
    for col in categorical
}
best_model.target = "target"
best_model.model_name = best_name

joblib.dump(best_model, "models/best_model.pkl")

test_data = X_test.copy()
test_data["target"] = y_test
test_data.to_csv("reports/test_data.csv", index=False)

print("\nBest model:", best_name)
print("Model saved successfully!")
