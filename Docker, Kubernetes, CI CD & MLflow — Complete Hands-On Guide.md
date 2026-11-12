# Docker, Kubernetes, CI/CD & MLflow — Complete Hands-On Guide

Nov 12, 2026 · @sarath

One repo covering containerization, orchestration, CI/CD automation, and experiment tracking — the full MLOps infrastructure layer.

## Overview

This guide covers four topics from a single repo (`docker-and-kubernetes`): containerizing an ML API with Docker, deploying it to Kubernetes with autoscaling, automating the build-test-train-validate pipeline with GitHub Actions, and tracking experiments with MLflow. All code builds on a single Iris classification model served via FastAPI.

**Prerequisites:** Python 3.11, Docker Desktop, Minikube + kubectl, Git, a GitHub account.

**Run order:**

1. Train model and build the FastAPI app
2. Dockerize it (single-stage → multi-stage → docker-compose with Redis)
3. Deploy to Kubernetes (Deployment → Service → ConfigMap/Secret → HPA)
4. Push to GitHub with CI/CD pipeline (lint → test → train → validate → build image)
5. Track experiments with MLflow (log runs → compare → register best model)

## Project Structure

```
docker-and-kubernetes/
├── .github/
│   └── workflows/
│       └── ml-ci.yml              ← GitHub Actions CI/CD pipeline
├── app/
│   ├── main.py                    ← FastAPI prediction server
│   ├── train.py                   ← model training script
│   └── model.pkl                  ← trained model (gitignored)
├── k8s/                           ← Kubernetes manifests
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── config.yaml                ← ConfigMap
│   ├── secret.yaml                ← Secret (gitignored)
│   └── hpa.yaml                   ← HorizontalPodAutoscaler
├── MLflow/
│   ├── mlflow_experiment.py       ← train + log 3 models
│   └── register_model.py          ← register best to registry
├── tests/
│   └── test_data_validation.py    ← pytest data validation
├── .dockerignore
├── .gitignore
├── Dockerfile                     ← single-stage build
├── Dockerfile.multistage          ← optimized multi-stage build
├── docker-compose.yml             ← API + Redis cache
├── requirements.txt
├── validate.py                    ← accuracy threshold check
└── ruff.toml                      ← linter config
```

**.gitignore should include:**

```
venv/
secret.yaml
model.pkl
metrics.json
mlruns/
mlartifacts/
__pycache__/
*.pyc
```

## Docker Hands-On (Day 5)

### Step 1: Train the model

**app/train.py:**

```python
import joblib
import json
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

joblib.dump(model, "app/model.pkl")
with open("metrics.json", "w") as f:
    json.dump({"accuracy": accuracy}, f)

print(f"Accuracy: {accuracy:.4f}")
```

### Step 2: FastAPI serving app

**app/main.py:**

```python
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Iris ML Prediction API")
model = joblib.load("model.pkl")


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
def predict(request: PredictRequest):
    features = np.array(request.features).reshape(1, -1)
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}
```

### Step 3: Dockerfile (single-stage)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`COPY requirements.txt` first — Docker caches this layer, so dependencies aren't reinstalled on every code change.

### Step 4: Build and run

```bash
docker build -t iris-ml-api:v1 .
docker run -p 8000:8000 iris-ml-api:v1
# Test: http://localhost:8000/health
# Test: http://localhost:8000/docs (Swagger UI)
```

### Step 5: Multi-stage Dockerfile

**Dockerfile.multistage:**

```dockerfile
# Build stage
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --target=/install -r requirements.txt

# Run stage
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /install /usr/local/lib/python3.11/site-packages/
COPY app/ .
RUN useradd -m appuser && chown -R appuser /app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build stage installs packages, run stage only copies the installed files. Result: smaller image, no pip/build tools in production, runs as non-root user.

```bash
docker build -f Dockerfile.multistage -t iris-ml-api:multistage .
docker images | findstr iris   # compare sizes
```

### Step 6: Docker Compose with Redis

**docker-compose.yml:**

```yaml
version: "3.8"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - redis
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

```bash
docker-compose up --build
# Both API and Redis start together
```

**requirements.txt:**

```
scikit-learn
joblib
numpy
fastapi
uvicorn
redis
pytest
ruff
```

## Kubernetes Hands-On (Day 6)

### Step 1: Start cluster and load image

```bash
minikube start --driver=docker
kubectl get nodes            # should show Ready
minikube image load iris-ml-api:v1
minikube image list | findstr iris   # verify
```

### Step 2: Deployment

**deployment.yaml:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: iris-deployment
  labels:
    app: iris
spec:
  replicas: 3
  selector:
    matchLabels:
      app: iris
  template:
    metadata:
      labels:
        app: iris
    spec:
      containers:
      - name: iris-container
        image: iris-ml-api:v1
        imagePullPolicy: Never
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: model-config
        - secretRef:
            name: model-secret
        resources:
          requests:
            cpu: "100m"
            memory: "128Mi"
          limits:
            cpu: "500m"
            memory: "512Mi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 15
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
```

`imagePullPolicy: Never` — use local image, don't pull from Docker Hub. `requests` = minimum guaranteed resources (scheduler uses these). `limits` = maximum allowed (pod killed if exceeded). `livenessProbe` — fails → pod restarted. `readinessProbe` — fails → traffic stopped to this pod.

```bash
kubectl apply -f deployment.yaml
kubectl get pods   # wait for 3x Running 1/1
```

### Step 3: Service

**service.yaml:**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: iris-service
spec:
  type: NodePort
  selector:
    app: iris
  ports:
  - port: 80
    targetPort: 8000
    nodePort: 30080
```

Service types: **ClusterIP** (internal only, default), **NodePort** (external access via node port 30000–32767), **LoadBalancer** (cloud only, real external IP). Traffic flow: `your machine → node:30080 → Service:80 → pod:8000`. Service load-balances across all 3 pods automatically.

```bash
kubectl apply -f service.yaml
minikube service iris-service --url   # get the URL
curl <url>/health
```

### Step 4: ConfigMap and Secret

**config.yaml:**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-config
data:
  MODEL_VERSION: "1.0"
  LOG_LEVEL: "INFO"
```

**secret.yaml:**

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: model-secret
type: Opaque
stringData:
  API_KEY: "my-secret-key-123"
```

ConfigMap = non-sensitive config. Secret = credentials (base64-encoded by K8s). Both injected as environment variables via `envFrom` in the Deployment. Never commit `secret.yaml` to Git — add to `.gitignore`. In production use a secret manager (Vault, AWS Secrets Manager, Azure Key Vault).

```bash
kubectl apply -f config.yaml
kubectl apply -f secret.yaml
kubectl apply -f deployment.yaml
kubectl exec -it <pod-name> -- env   # verify MODEL_VERSION, API_KEY
```

### Step 5: HPA (Horizontal Pod Autoscaler)

**hpa.yaml:**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: iris-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: iris-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 50
```

HPA calculates: `desired = current × (actual_cpu / target_cpu)`. If 3 pods at 80% CPU: `3 × (80/50) = 5 pods`. Scales up immediately, scales down after 5 minutes (prevents flapping). Once HPA is active, it owns the replica count — the Deployment's `replicas: 3` is just the starting point.

```bash
minikube addons enable metrics-server
kubectl apply -f hpa.yaml
kubectl get hpa         # watch scaling
kubectl top nodes       # check resource usage
```

### Cleanup

```bash
kubectl delete -f .     # delete all K8s resources
minikube stop
```

## CI/CD with GitHub Actions (Day 8)

### The workflow

**.github/workflows/ml-ci.yml:**

```yaml
name: ML CI Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  ml-pipeline:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Lint with ruff
        run: ruff check .

      - name: Run tests
        run: pytest tests/ -v

      - name: Train model
        run: python app/train.py

      - name: Check model accuracy
        run: python validate.py

      - name: Build Docker image
        run: docker build -t iris-ml-api:${{ github.sha }} .

      - name: Log in to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}

      - name: Push Docker image
        run: |
          docker tag iris-ml-api:${{ github.sha }} ${{ secrets.DOCKER_USERNAME }}/iris-ml-api:latest
          docker push ${{ secrets.DOCKER_USERNAME }}/iris-ml-api:latest
```

Triggers on every push to `main`. Each step depends on the previous — if any fails, the pipeline stops. The Docker push steps require `DOCKER_USERNAME` and `DOCKER_PASSWORD` secrets in GitHub repo Settings → Secrets → Actions.

### Data validation tests

**tests/test\_data\_validation.py:**

```python
from sklearn.datasets import load_iris
import numpy as np


def load_data():
    X, y = load_iris(return_X_y=True)
    return X, y


def test_no_nulls():
    X, y = load_data()
    assert not np.isnan(X).any()
    assert not np.isnan(y).any()


def test_feature_count():
    X, _ = load_data()
    assert X.shape[1] == 4


def test_label_values():
    _, y = load_data()
    assert set(np.unique(y)) == {0, 1, 2}


def test_no_empty_dataset():
    X, y = load_data()
    assert len(X) > 0
    assert len(X) == len(y)
```

Runs before training — catches bad data before you waste compute.

### Accuracy validation

**validate.py:**

```python
import json
import sys

THRESHOLD = 0.90

with open("metrics.json") as f:
    metrics = json.load(f)

accuracy = metrics["accuracy"]

if accuracy >= THRESHOLD:
    print(f"Accuracy {accuracy:.4f} >= {THRESHOLD}. Ready to deploy!")
else:
    print(f"Accuracy {accuracy:.4f} < {THRESHOLD}. Not good enough.")
    sys.exit(1)
```

`sys.exit(1)` makes the pipeline fail — a bad model never gets built into an image.

### Linter config

**ruff.toml:**

```toml
[lint]
ignore = ["BLE001"]
```

The full pipeline flow: `code push → lint → test data → train model → validate accuracy → build image → push to registry`. This is CI/CD/CT for ML — the "CT" (Continuous Training) is the train + validate steps that regular software pipelines don't have.

## MLflow Hands-On (Day 9)

### Step 1: Train and log experiments

**MLflow/mlflow\_experiment.py:**

```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("iris-model-comparison")

# Run 1: Logistic Regression
with mlflow.start_run(run_name="logistic-regression"):
    lr = LogisticRegression(max_iter=200, C=1.0)
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", 200)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(lr, name="model")

# Run 2: Random Forest
with mlflow.start_run(run_name="random-forest"):
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(
        rf, name="model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

# Run 3: Random Forest v2 (different params)
with mlflow.start_run(run_name="random-forest-v2"):
    rf2 = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf2.fit(X_train, y_train)
    preds = rf2.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(
        rf2, name="model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

print("All runs logged. Run 'mlflow ui' to view.")
```

Tree-based models need `skops_trusted_types=["sklearn.tree._tree.Tree"]` in newer MLflow versions. Logistic Regression doesn't need it.

```bash
pip install mlflow
python MLflow/mlflow_experiment.py
mlflow ui   # open http://127.0.0.1:5000
```

### Step 2: Register the best model

**MLflow/register\_model.py:**

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

experiment = client.get_experiment_by_name("iris-model-comparison")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"],
    max_results=1
)

best_run = runs[0]
print(f"Best run: {best_run.info.run_id}")
print(f"Accuracy: {best_run.data.metrics['accuracy']:.4f}")

model_uri = f"runs:/{best_run.info.run_id}/model"
result = mlflow.register_model(model_uri, "iris-classifier")
print(f"Registered: {result.name}, version: {result.version}")

client.transition_model_version_stage(
    name="iris-classifier",
    version=result.version,
    stage="Staging"
)
print("Moved to Staging")

client.transition_model_version_stage(
    name="iris-classifier",
    version=result.version,
    stage="Production"
)
print("Moved to Production")
```

`model_uri = f"runs:/{run_id}/model"` — points to the model artifact inside a specific run. `register_model` creates a versioned entry in the Model Registry. `transition_model_version_stage` promotes through: None → Staging → Production.

### What to check in the MLflow UI

- **Runs tab:** all 3 runs with params, metrics. Select all → Compare
- **Inside a run:** Parameters, Metrics, Artifacts (MLmodel, model.pkl, conda.yaml, requirements.txt)
- **Chart view:** visual comparison of metrics across runs
- **Model Registry:** iris-classifier with version and Production stage
- **Failed runs:** stay logged for debugging and audit trail

## Key Interview Points

**Docker:**

- "Why multi-stage builds?" — Smaller image, no build tools in production, faster pulls.
- "What's in your Dockerfile?" — Copy requirements first (layer caching), copy code, non-root user, expose port, uvicorn entrypoint.
- "How do you handle secrets in Docker?" — Environment variables at runtime, never baked into the image. Use docker-compose `env_file` or Kubernetes Secrets.

**Kubernetes:**

- "Explain Deployment vs Service" — Deployment manages pods (replicas, rolling updates). Service provides a stable network endpoint to access those pods.
- "Liveness vs Readiness probe?" — Liveness: is the pod alive? (fail → restart). Readiness: can it handle traffic? (fail → stop routing, don't restart).
- "Requests vs Limits?" — Requests = minimum guaranteed (scheduler uses these). Limits = maximum allowed (exceeding memory → OOMKilled, exceeding CPU → throttled).
- "How does HPA work?" — Watches CPU/memory metrics, calculates desired replicas using the formula `current × (actual/target)`, scales between min and max.
- "ClusterIP vs NodePort vs LoadBalancer?" — ClusterIP: internal only. NodePort: external via high port on every node. LoadBalancer: cloud-managed external IP.

**CI/CD:**

- "What's different about ML CI/CD?" — Regular CI/CD tests code. ML CI/CD also validates data quality (pytest on data) and model quality (accuracy threshold check). That's the CT (Continuous Training) part.
- "How do you prevent bad models from shipping?" — `validate.py` checks accuracy against a threshold. `sys.exit(1)` fails the pipeline if the model underperforms. The Docker image is only built after validation passes.
- "Why GitHub Actions?" — Free for public repos, YAML-based, direct GitHub integration, marketplace of pre-built actions.

**MLflow:**

- "How do you track experiments?" — MLflow logs params, metrics, and model artifacts per run. Compare runs in the UI. Programmatically search for the best run by metric.
- "What's the Model Registry for?" — Central catalog of all production models. Version-tracked: v1, v2, v3. Stage management: None → Staging → Production. Serving code loads `models:/name/Production` — promoting a new version updates serving without code changes.
- "autolog vs manual logging?" — autolog is convenient for exploration. Manual logging gives full control and avoids compatibility issues in production environments (especially Azure ML).
