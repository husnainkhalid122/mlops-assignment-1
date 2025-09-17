import os
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Function to evaluate models
def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro"),
        "f1": f1_score(y_true, y_pred, average="macro")
    }

def main():
    # === DATASET ===
    iris = load_iris(as_frame=True)
    df = iris.frame
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/iris.csv", index=False)   # Save dataset in /data

    X = df.drop("target", axis=1)
    y = df["target"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Models
    models = {
        "logistic_regression": LogisticRegression(max_iter=200),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "svm": SVC(kernel="rbf", C=1.0)
    }

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate
        metrics = evaluate(y_test, y_pred)
        print(f"{name} metrics:", metrics)
        print(classification_report(y_test, y_pred))

        # Save metrics
        metrics_file = os.path.join("results", f"{name}_metrics.txt")
        with open(metrics_file, "w") as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")

        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        cm_path = os.path.join("results", f"{name}_cm.png")
        plt.savefig(cm_path)
        plt.close()

        # Save model
        file_path = os.path.join("models", f"{name}_v1.pkl")
        joblib.dump(model, file_path)
        print(f"Saved {name} model at {file_path}")

if __name__ == "__main__":
    main()

