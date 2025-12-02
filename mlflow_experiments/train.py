import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


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

    # 2. Define models to compare
    models = [
        (
            "logistic_regression",
            LogisticRegression(max_iter=500, solver="liblinear"),
            {"max_iter": 500, "solver": "liblinear"},
        ),
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                random_state=42
            ),
            {"n_estimators": 200, "max_depth": 6},
        ),
        (
            "gradient_boosting",
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            ),
            {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3},
        ),
    ]

    # 3. Tell MLflow which experiment we’re working in
    mlflow.set_experiment(EXPERIMENT_NAME)

    best_f1 = -1.0
    best_model = None
    best_model_name = None

    # 4. Loop over models, create one MLflow run per model
    for model_name, model, params in models:
        with mlflow.start_run(run_name=model_name):
            # Train
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Log params & metrics to MLflow
            mlflow.log_param("model_name", model_name)
            for p_name, p_value in params.items():
                mlflow.log_param(p_name, p_value)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)

            # Log the model as an artifact
            mlflow.sklearn.log_model(model, artifact_path="model")

            print(f"{model_name}: accuracy={acc:.4f}, f1={f1:.4f}")

            # Track best model based on F1 score
            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_model_name = model_name

    # 5. Save the best model to service/model.pkl for deployment later
    if best_model is not None:
        os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)
        joblib.dump(best_model, BEST_MODEL_PATH)
        print(f"\nBest model: {best_model_name} (f1={best_f1:.4f})")
        print(f"Saved best model to: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()  