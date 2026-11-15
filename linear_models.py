from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)
import numpy as np

def train_linear_regression(data: RegressionInput) -> RegressionResult:
    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=float).reshape(-1)
    X_test = np.asarray(data.test_features, dtype=float)

    # Feature scaling
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    weights = np.zeros(X_train.shape[1])
    bias = 0.0
    loss_history = []

    for epoch in range(data.epochs):
        predictions = X_train @ weights + bias
        errors = predictions - y_train

        dw = (2 / len(y_train)) * (X_train.T @ errors)
        db = (2 / len(y_train)) * np.sum(errors)

        weights -= data.learning_rate * dw
        bias -= data.learning_rate * db

        updated_predictions = X_train @ weights + bias
        mse = np.mean((updated_predictions - y_train) ** 2)
        loss_history.append(float(mse))

    test_predictions = X_test @ weights + bias

    return RegressionResult(
        weights=weights,
        bias=float(bias),
        loss_history=loss_history,
        predictions=test_predictions
    )


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
    raise NotImplementedError


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split.

    После запуска запишите реальные результаты в experiment_results.json:
    {"linear_regression":{"test_mse": число},
     "classification":{"own_logistic": {"roc_auc": ..., "accuracy": ..., "precision": ..., "recall": ..., "f1": ...},
                       "logistic_regression": {...}, "decision_tree": {...}, "knn": {...}}}.
    """
    raise NotImplementedError
