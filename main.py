# main.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from PIL import Image
import io

from model import predict_value, predict_image, load_model

app = FastAPI(title="API Engine Starter", version="0.1.0")

# Serve static demo UI
app.mount("/static", StaticFiles(directory="static"), name="static")

class InputData(BaseModel):
    x1: float
    x2: float

class BatchPayload(BaseModel):
    items: List[InputData]

@app.get("/")
def home():
    return {"service": "API Engine Starter", "version": "0.1.0"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: InputData):
    try:
        out = predict_value(data.x1, data.x2)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"prediction": out}

@app.post("/predict/batch")
def predict_batch(payload: BatchPayload):
    try:
        preds = []
        for item in payload.items:
            preds.append(predict_value(item.x1, item.x2))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"predictions": preds}

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image and get inference. This function uses PIL to read the image
    and calls predict_image (user to implement for their model).
    """
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        result = predict_image(img)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse({"result": result})
