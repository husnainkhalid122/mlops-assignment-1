# MLOps Assignment 1

## Student Info
- **Name:** Husnain Khalid
- **Roll No:** 22F-3190

---

## 1. Problem Statement & Dataset Description
The task of this assignment was to:
1. Create a GitHub repository with proper folder structure.
2. Train at least three machine learning models on a dataset.
3. Compare models on evaluation metrics (accuracy, precision, recall, F1-score).
4. Use MLflow for experiment tracking, logging parameters, metrics, and artifacts.
5. Register the best-performing model in MLflow Model Registry.
6. Monitor the registered model by logging new metrics on unseen data.
7. Document all steps and push to GitHub.

### Dataset
We used the **Iris dataset** from `sklearn.datasets`.  
- Features: sepal length, sepal width, petal length, petal width.  
- Target: three Iris flower species (`setosa`, `versicolor`, `virginica`).  

---

## 2. Model Selection & Comparison
Three models were trained:
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)

Evaluation metrics used:
- Accuracy
- Precision
- Recall
- F1-Score

The models were compared, and their performance was logged in MLflow. The best-performing model was **Random Forest**.

Models were saved in `/models` folder as `.pkl` files.

---

## 3. MLflow Logging (Screenshots)
MLflow was used for experiment tracking. The following screenshots were generated:
1. **Experiment Page:** showing all 3 runs and metrics.  
2. **Single Run Detail:** showing parameters, metrics, and artifacts.  
3. **Confusion Matrix Artifact:** logged for each model.  

*(Screenshots attached in Word report submission)*

---

## 4. Model Registration (Screenshots)
The best model was registered into the **MLflow Model Registry**.  
- Model name: `mlops_assignment_model`  
- Version 1 was promoted to **Staging** and later to **Production**.  

*(Screenshots attached in Word report submission)*

---

## 5. Monitoring
A monitoring script (`src/monitor_model.py`) was created to simulate new data evaluation.  
- The model was re-tested on new splits of the Iris dataset.  
- New metrics were logged to MLflow as a **monitoring run**.  

---

## 6. Instructions to Run the Code

### Prerequisites
- Python 3.9+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
