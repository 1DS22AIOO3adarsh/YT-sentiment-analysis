from src.data_loader import load_data
from src.preprocessor import preprocess_dataframe
from featurizers import tfidf_featurizer
from models.logistic import get_model
from src.trainer import train_and_log
import pandas as pd
import joblib
import os


# Load and preprocess data
X_train, X_test, y_train, y_test = load_data("data/youtube_comments.csv")
train_df = pd.DataFrame({"Comment": X_train, "Sentiment": y_train})
test_df = pd.DataFrame({"Comment": X_test, "Sentiment": y_test})

train_df = preprocess_dataframe(train_df)
test_df = preprocess_dataframe(test_df)

X_train = train_df["Comment"]
X_test = test_df["Comment"]
y_train = train_df["Sentiment"]
y_test = test_df["Sentiment"]

# Use only best featurizer and model
vectorizer = tfidf_featurizer.get_vectorizer()
model = get_model()  # This returns a LogisticRegression instance

train_and_log("tfidf", "logreg", X_train, y_train, X_test, y_test, vectorizer, model)

os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/logreg_model.pkl")
joblib.dump(vectorizer, "artifacts/tfidf_vectorizer.pkl")