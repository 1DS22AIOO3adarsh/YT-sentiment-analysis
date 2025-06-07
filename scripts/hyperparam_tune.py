import mlflow
import mlflow.sklearn as mlflow_sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from src.data_loader import load_data  # adjust this import as per your project

def tune_logistic_regression(X_train, y_train, X_test, y_test):
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    param_grid = {
        'C': [0.01, 0.1, 1, 10],  # regularization strength inverse
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [100, 200]
    }

    lr = LogisticRegression()

    grid_search = GridSearchCV(lr, param_grid, scoring='f1_weighted', cv=5, verbose=1)
    grid_search.fit(X_train_vec, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    mlflow.set_experiment("SentimentAnalysisExperiment")

    with mlflow.start_run(run_name="logreg_hyperparam_tuning"):
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", float(acc))
        mlflow.log_metric("f1_score", float(f1))
        mlflow_sklearn.log_model(best_model, "model")
        mlflow.log_param("vectorizer", "tfidf")

    print(f"Best params: {grid_search.best_params_}")
    print(f"Accuracy: {acc:.4f}, F1-score: {f1:.4f}")

def main():
    X_train, X_test, y_train, y_test = load_data('data/youtube_comments.csv')
    tune_logistic_regression(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
