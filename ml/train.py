# ml/train.py
from sklearn.linear_model import LinearRegression
import pandas as pd
import joblib
import os

os.makedirs(os.path.dirname(__file__), exist_ok=True)

def train_and_save():
    # Simple demo data: y = 3*x1 + 0.6*x2 (rough)
    data = {
        "x1": [1, 2, 3, 4, 5],
        "x2": [10, 20, 30, 40, 50],
        "y":  [15, 30, 45, 60, 75]
    }

    df = pd.DataFrame(data)
    X = df[["x1", "x2"]]
    y = df["y"]

    model = LinearRegression()
    model.fit(X, y)

    out_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    joblib.dump(model, out_path)
    print(f"Model trained and saved to {out_path}")

if __name__ == "__main__":
    train_and_save()
