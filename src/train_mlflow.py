import sys
sys.stdout.reconfigure(encoding='utf-8')
# src/train_mlflow.py
import os
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

def evaluate(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    }

def plot_and_save_cm(cm, classes, out_path):
    import matplotlib
    matplotlib.use('Agg')  # headless backend
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def main():
    # Optional: set experiment name
    mlflow.set_experiment("mlops_assignment_1")

    iris = load_iris()
    X, y = iris.data, iris.target
    class_names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "logistic_regression": LogisticRegression(max_iter=200),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "svm": SVC(kernel="rbf", C=1.0, probability=True)
    }

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    for name, clf in models.items():
        # Create pipeline so scaling is included in saved model
        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

        # Start MLflow run
        with mlflow.start_run(run_name=name) as run:
            run_id = run.info.run_id

            # Log model hyperparameters (example)
            params = {}
            if name == "random_forest":
                params = {"n_estimators": clf.get_params().get("n_estimators")}
            elif name == "logistic_regression":
                params = {"max_iter": clf.get_params().get("max_iter")}
            elif name == "svm":
                params = {"kernel": clf.get_params().get("kernel"), "C": clf.get_params().get("C")}
            mlflow.log_params(params)

            # Train
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)

            # Evaluate
            metrics = evaluate(y_test, preds)
            mlflow.log_metrics(metrics)

            # Save confusion matrix to results and log as artifact
            cm = confusion_matrix(y_test, preds)
            cm_file = os.path.join("results", f"{name}_cm.png")
            plot_and_save_cm(cm, class_names, cm_file)
            mlflow.log_artifact(cm_file, artifact_path="confusion_matrices")

            # Save local model file (in /models)
            local_model_path = os.path.join("models", f"{name}.pkl")
            joblib.dump(pipeline, local_model_path)
            # Log the saved file as artifact
            mlflow.log_artifact(local_model_path, artifact_path="saved_models")

            # Also log MLflow model (for registry / deployment)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            print(f"Run {run_id} logged for {name} with metrics: {metrics}")

if __name__ == "__main__":
    main()
