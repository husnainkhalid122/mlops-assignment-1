# src/register_model.py
import mlflow
from mlflow.tracking import MlflowClient

# set tracking URI (same as when running training)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Replace with the BEST run id you want to register (copy from MLflow UI)
BEST_RUN_ID = "593dd10caf1e476abdd9ccdedf3cec42"
MODEL_NAME = "mlops_assignment_model"

client = MlflowClient()
model_uri = f"runs:/{BEST_RUN_ID}/model"
mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
print("Registered model:", mv.name, "version:", mv.version)

# Optionally transition version to "Staging"
client.transition_model_version_stage(
    name=mv.name,
    version=mv.version,
    stage="Staging"
)
print("Moved model to Staging")
