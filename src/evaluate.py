import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

def save_clean_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    img = Image.open(path)
    clean_img = Image.frombytes(img.mode, img.size, img.tobytes())
    clean_img.save(path)

os.makedirs("reports", exist_ok=True)
model = joblib.load("models/best_model.pkl")
test_data = pd.read_csv("reports/test_data.csv")

X_test = test_data.drop(columns=["target"])
y_test = test_data["target"]
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\n========== MODEL EVALUATION ==========")
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions, zero_division=0))
print("Recall:", recall_score(y_test, predictions, zero_division=0))
print("F1 Score:", f1_score(y_test, predictions, zero_division=0))
print("ROC AUC:", roc_auc_score(y_test, probabilities))
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)

fig1 = plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Disease", "Disease"],
    yticklabels=["No Disease", "Disease"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Heart Disease Prediction - Confusion Matrix")
save_clean_figure(fig1, "reports/confusion_matrix.png")

fpr, tpr, thresholds = roc_curve(y_test, probabilities)
fig2 = plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label="Model ROC Curve")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
save_clean_figure(fig2, "reports/roc_curve.png")

print("\nEvaluation reports saved in reports/ folder.")
