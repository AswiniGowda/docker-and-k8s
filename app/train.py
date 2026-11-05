
import json

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# 1. Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Initialize Logistic Regression
model = LogisticRegression(max_iter=200)

# 4. Train model
model.fit(X_train, y_train)

# 5. Evaluate model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Training Completed!")
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=iris.target_names))

# 6. Save trained model
joblib.dump(model, "app/model.pkl")

print("\nModel saved successfully as app/model.pkl")

with open("metrics.json", "w") as f:
    json.dump({"accuracy": accuracy}, f)
