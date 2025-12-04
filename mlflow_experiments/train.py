import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC  # NEW: Support Vector Machine


# -------- SETTINGS --------
EXPERIMENT_NAME = "mlops_demo_breast_cancer"
BEST_MODEL_PATH = os.path.join("service", "model.pkl")  # where the best model will be saved


def main():
    # 1. Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 2. Define model families + small param grids
    # Each entry: model_name -> {"estimator": Class, "param_grid": [dict, dict, ...]}
    model_configs = {
        "logistic_regression": {
            "estimator": LogisticRegression,
            "param_grid": [
                {"max_iter": 500, "solver": "liblinear", "C": 1.0},
                {"max_iter": 1000, "solver": "liblinear", "C": 0.5},
            ],
        },
        "random_forest": {
            "estimator": RandomForestClassifier,
            "param_grid": [
                {"n_estimators": 200, "max_depth": 6, "random_state": 42},
                {"n_estimators": 400, "max_depth": 8, "random_state": 42},
            ],
        },
        "gradient_boosting": {
            "estimator": GradientBoostingClassifier,
            "param_grid": [
                {
                    "n_estimators": 200,
                    "learning_rate": 0.05,
                    "max_depth": 3,
                    "random_state": 42,
                },
                {
                    "n_estimators": 300,
                    "learning_rate": 0.05,
                    "max_depth": 2,
                    "random_state": 42,
                },
            ],
        },
        "svm_rbf": {
            "estimator": SVC,
            "param_grid": [
                {"kernel": "rbf", "C": 1.0, "gamma": "scale", "probability": True, "random_state": 42},
                {"kernel": "rbf", "C": 0.5, "gamma": "scale", "probability": True, "random_state": 42},
            ],
        },
    }

    # 3. Tell MLflow which experiment we’re working in
    mlflow.set_experiment(EXPERIMENT_NAME)

    best_f1 = -1.0
    best_model = None
    best_model_label = None
    best_params = None

    # 4. Loop over model families + param configs
    for model_name, cfg in model_configs.items():
        EstimatorClass = cfg["estimator"]
        param_grid = cfg["param_grid"]

        for i, params in enumerate(param_grid):
            run_label = f"{model_name}_config_{i}"

            with mlflow.start_run(run_name=run_label):
                # ---- Train ----
                model = EstimatorClass(**params)
                model.fit(X_train, y_train)

                # ---- Evaluate ----
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                # ---- Log params & metrics to MLflow ----
                mlflow.log_param("model_family", model_name)
                mlflow.log_params(params)  # logs all hyperparams in this config
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1_score", f1)

                # Log the model artifact for this specific config
                mlflow.sklearn.log_model(model, artifact_path="model")

                print(f"{run_label}: accuracy={acc:.4f}, f1={f1:.4f}")

                # ---- Track best model across ALL configs ----
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model
                    best_model_label = run_label
                    best_params = params

    # 5. Save the best model to service/model.pkl for deployment later
    if best_model is not None:
        os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)
        joblib.dump(best_model, BEST_MODEL_PATH)
        print(f"\nBest config: {best_model_label} (f1={best_f1:.4f})")
        print(f"Best hyperparameters: {best_params}")
        print(f"Saved best model to: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()