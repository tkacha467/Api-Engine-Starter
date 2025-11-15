# model.py
import os
import joblib
from typing import Any
from PIL import Image
import numpy as np

MODEL_PATH = os.path.join("ml", "model.pkl")

# Lazy-load model to avoid blocking app startup if model missing
_model = None

def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run `python ml/train.py` first.")
        _model = joblib.load(MODEL_PATH)
    return _model

def predict_value(x1: float, x2: float) -> float:
    """
    Simple wrapper for tabular models that accept two features.
    """
    model = load_model()
    pred = model.predict([[x1, x2]])
    return float(pred[0])

# --- image helpers (for image models) ---
def preprocess_image(img: Image.Image, size=(224, 224)) -> np.ndarray:
    """
    Resize and normalize PIL image to numpy array expected by model.
    Adjust for your model's expected preprocessing.
    """
    img = img.convert("RGB")
    img = img.resize(size)
    arr = np.array(img).astype("float32") / 255.0
    # expand dims to (1, H, W, C)
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(img: Image.Image) -> Any:
    """
    Example placeholder. If you have an image model, adapt this to your
    model's predict method (e.g., Keras/Torch outputs).
    For demo purposes we'll call a tabular model with dummy inputs.
    """
    # If you actually have a deep learning model, replace this logic.
    # For demo: return a dummy label
    model = load_model()  # if you used a DL model, load it differently
    # placeholder: return shape info
    arr = preprocess_image(img, size=(64, 64))
    return {"shape": arr.shape}
