import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from grader_contracts.linear_models import (
    RegressionInput,
    ClassificationInput,
)

from linear_models import (
    train_linear_regression,
    train_logistic_classifier,
    compare_classical_models,
)


# 1. Linear Regression
student = pd.read_csv("Student_Performance.csv")

student["Extracurricular Activities"] = (
    student["Extracurricular Activities"].map({"Yes": 1, "No": 0})
)

X_reg = student.drop(columns=["Performance Index"]).to_numpy(dtype=float)
y_reg = student["Performance Index"].to_numpy(dtype=float)

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg_input = RegressionInput(
    train_features=X_train,
    train_target=y_train,
    test_features=X_test,
    test_target=y_test,
)

reg_result = train_linear_regression(reg_input)

test_mse = float(np.mean((reg_result.predictions - y_test) ** 2))


# 2. Logistic Regression
german = pd.read_csv("german.csv", sep=";")

X_cls = german.iloc[:, 1:].to_numpy(dtype=float)
y_cls = german.iloc[:, 0].to_numpy(dtype=int)

X_train, X_test, y_train, y_test = train_test_split(
    X_cls,
    y_cls,
    test_size=0.2,
    random_state=42,
    stratify=y_cls,
)

cls_input = ClassificationInput(
    train_features=X_train,
    train_target=y_train,
    test_features=X_test,
    test_target=y_test,
)

own_result = train_logistic_classifier(cls_input)


def calculate_metrics(y_true, predictions, probabilities):
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


own_metrics = calculate_metrics(
    y_test,
    own_result.predictions,
    own_result.probabilities,
)


# 3. Classical models on the same split
baseline_result = compare_classical_models(cls_input)


# 4. Save measured results
results = {
    "linear_regression": {
        "test_mse": test_mse
    },
    "classification": {
        "own_logistic": own_metrics,
        "logistic_regression": asdict(
            baseline_result.logistic_regression
        ),
        "decision_tree": asdict(
            baseline_result.decision_tree
        ),
        "knn": asdict(baseline_result.knn),
    },
}

with open("experiment_results.json", "w", encoding="utf-8") as file:
    json.dump(results, file, indent=2, ensure_ascii=False)

print(json.dumps(results, indent=2, ensure_ascii=False))
print("\nResults saved to experiment_results.json")
