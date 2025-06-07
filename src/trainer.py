import mlflow
import mlflow.sklearn as mlflow_sklearn
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from mlflow.models.signature import infer_signature

def train_and_log(vec_name, model_name, X_train, y_train, X_test, y_test, vectorizer, model):
    # Vectorize data
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    mlflow.set_experiment("SentimentAnalysisExperiment")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"{vec_name}_{model_name}"):
        # Fit model
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Log params and metrics
        mlflow.log_param("vectorizer", vec_name)
        mlflow.log_param("model", model_name)
        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("f1_score", float(f1))

        # Create input example and signature for better model tracking
        input_example = pd.DataFrame(X_test_vec[:1].toarray())  # Sparse to dense
        signature = infer_signature(X_test_vec, y_pred)

        # Log model with signature and example
        mlflow_sklearn.log_model(
            model,
            "model",
            input_example=input_example,
            signature=signature
        )

        print(f"Logged run: {vec_name} + {model_name} | acc={acc:.4f}, f1={f1:.4f}")
        return model, {"accuracy": acc, "f1_score": f1}
