import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set experiment name
mlflow.set_experiment("iris-model-comparison")

# --- Run 1: Logistic Regression ---
with mlflow.start_run(run_name="logistic-regression"):
    lr = LogisticRegression(max_iter=200, C=1.0)
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 200)
    mlflow.log_param("C", 1.0)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(lr, "model")

    print(f"Logistic Regression - Accuracy: {acc:.4f}, F1: {f1:.4f}")

# --- Run 2: Random Forest ---
with mlflow.start_run(run_name="random-forest"):
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(
        rf, name ="model",
        skops_trusted_types=["sklearn.tree._tree.Tree"])

    print(f"Random Forest - Accuracy: {acc:.4f}, F1: {f1:.4f}")

# --- Run 3: Random Forest (different params) ---
with mlflow.start_run(run_name="random-forest-v2"):
    rf2 = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf2.fit(X_train, y_train)
    preds = rf2.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(
        rf2, name="model",
        skops_trusted_types=["sklearn.tree._tree.Tree"])

    print(f"Random Forest v2 - Accuracy: {acc:.4f}, F1: {f1:.4f}")

print("\nAll runs logged. Run 'mlflow ui' to view.")