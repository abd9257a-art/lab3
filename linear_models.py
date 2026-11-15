from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from grader_contracts.linear_models import ClassificationMetrics
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
        logits = X_train @ weights + bias
        logits = np.clip(logits, -500, 500)
        probabilities = 1 / (1 + np.exp(-logits))

        errors = probabilities - y_train

        dw = (X_train.T @ errors) / len(y_train)
        db = np.mean(errors)

        weights -= data.learning_rate * dw
        bias -= data.learning_rate * db

        updated_logits = np.clip(X_train @ weights + bias, -500, 500)
        updated_probabilities = 1 / (1 + np.exp(-updated_logits))

        epsilon = 1e-15
        loss = -np.mean(
            y_train * np.log(updated_probabilities + epsilon)
            + (1 - y_train) * np.log(1 - updated_probabilities + epsilon)
        )
        loss_history.append(float(loss))

    test_logits = np.clip(X_test @ weights + bias, -500, 500)
    test_probabilities = 1 / (1 + np.exp(-test_logits))
    predictions = (test_probabilities >= 0.5).astype(int)

    return ClassificationResult(
        weights=weights,
        bias=float(bias),
        loss_history=loss_history,
        probabilities=test_probabilities,
        predictions=predictions
    )



def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    X_train = np.asarray(data.train_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=int).reshape(-1)
    X_test = np.asarray(data.test_features, dtype=float)
    y_test = np.asarray(data.test_target, dtype=int).reshape(-1)

    def calculate_metrics(model):
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        return ClassificationMetrics(
            roc_auc=float(roc_auc_score(y_test, probabilities)),
            accuracy=float(accuracy_score(y_test, predictions)),
            precision=float(precision_score(y_test, predictions, zero_division=0)),
            recall=float(recall_score(y_test, predictions, zero_division=0)),
            f1=float(f1_score(y_test, predictions, zero_division=0)),
        )

    logistic_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42)
    )
    logistic_model.fit(X_train, y_train)

    tree_model = DecisionTreeClassifier(random_state=42)
    tree_model.fit(X_train, y_train)

    knn_model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    )
    knn_model.fit(X_train, y_train)

    return BaselineComparisonResult(
        logistic_regression=calculate_metrics(logistic_model),
        decision_tree=calculate_metrics(tree_model),
        knn=calculate_metrics(knn_model),
    )
