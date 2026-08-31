import io
import os
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

MODEL_DIR = "shadmanArko/dhaka-kacchi-dish-classifier"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "..", "frontend")
static_dir = os.path.normpath(static_dir)

app = FastAPI(title="Dhaka Kacchi Dish Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
model = AutoModelForImageClassification.from_pretrained(MODEL_DIR)
model.eval()


@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]

    predicted_idx = probs.argmax().item()
    predicted_label = model.config.id2label[predicted_idx]
    confidence = probs[predicted_idx].item()

    return {
        "prediction": predicted_label,
        "confidence": round(confidence, 4),
    }
