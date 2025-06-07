from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("artifacts/logreg_model.pkl")
vectorizer = joblib.load("artifacts/tfidf_vectorizer.pkl")

class TextInput(BaseModel):
    text: str

@app.post("/predict/")
def predict_sentiment(input: TextInput):
    vect_text = vectorizer.transform([input.text])
    prediction = model.predict(vect_text)[0]
    proba = model.predict_proba(vect_text)[0]

    label_map = {
    0: "Negative",
    1: "Positive",
    2: "Neutral"  
}

    confidence = round(float(max(proba)) * 100, 2)  # Convert to percentage

    return {
        "sentiment": label_map[int(prediction)],
        "confidence": f"{confidence} %"
    }
