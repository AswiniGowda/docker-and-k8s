import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Find the best run by accuracy
experiment = client.get_experiment_by_name("iris-model-comparison")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"],
    max_results=1
)

best_run = runs[0]
print(f"Best run: {best_run.info.run_id}")
print(f"Accuracy: {best_run.data.metrics['accuracy']:.4f}")

# Register the model
model_uri = f"runs:/{best_run.info.run_id}/model"
result = mlflow.register_model(model_uri, "iris-classifier")
print(f"Model registered: {result.name}, version: {result.version}")

# Move to Staging
client.transition_model_version_stage(
    name="iris-classifier",
    version=result.version,
    stage="Staging"
)
print("Moved to Staging")

# Move to Production
client.transition_model_version_stage(
    name="iris-classifier",
    version=result.version,
    stage="Production"
)
print("Moved to Production")