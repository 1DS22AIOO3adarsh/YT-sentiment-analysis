import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, List
from sklearn.preprocessing import LabelEncoder

def load_data(filepath: str, test_size: float = 0.2, random_state: int = 42):
    df = pd.read_csv(filepath)

    # Check required columns
    if 'Comment' not in df.columns or 'Sentiment' not in df.columns:
        raise ValueError("CSV must contain 'Comment' and 'Sentiment' columns.")

    # Encode Sentiment to numeric
    label_encoder = LabelEncoder()
    df['Sentiment'] = label_encoder.fit_transform(df['Sentiment'])

    X = df['Comment'].astype(str).tolist()
    y = df['Sentiment'].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test
