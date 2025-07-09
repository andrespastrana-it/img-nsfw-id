from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from typing import List
from PIL import Image
import opennsfw2 as n2
import numpy as np
import requests
import io
import tempfile
import os
import cv2

# --- Classification Thresholds ---
NSFW_THRESHOLD = 0.8
MATURE_THRESHOLD = 0.5
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt','gender_net.caffemodel')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
age_list = ['(0-2)', '(4-6)', '(8-18)', '(19-25)',
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# --- Helper ---
def classify_score(score: float) -> str:
    if score > NSFW_THRESHOLD:
        return "nsfw"
    elif score > MATURE_THRESHOLD:
        return "mature"
    else:
        return "sfw"

# --- App ---
app = FastAPI()

# -------------------------------------
# Health and Version Info
# -------------------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/version")
def version():
    return {
        "service": "OpenNSFW2 API",
        "version": "1.0.0",
        "model": "OpenNSFW2",
        "backend": "TensorFlow",
        "thresholds": {
            "nsfw": NSFW_THRESHOLD,
            "mature": MATURE_THRESHOLD
        }
    }

# -------------------------------------
# 1. Classify Image from URL
# -------------------------------------
class ImageURLRequest(BaseModel):
    url: HttpUrl

@app.post("/analyze/image")
async def analyze_image_by_url(
    request: ImageURLRequest,
    threshold: float = Query(NSFW_THRESHOLD, ge=0.0, le=1.0)
):
    try:
        response = requests.get(request.url, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        np_image = np.array(image)
        np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load or decode image: {str(e)}")

    predicted_age = "Unknown"

    try:
        gray = cv2.cvtColor(np_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            face_img = np_image[y:y+h, x:x+w]
            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]

            age_net.setInput(blob)
            age_preds = age_net.forward()
            predicted_age = age_list[age_preds[0].argmax()]
            break  # Only process the first face for simplicity

        # NSFW check
        nsfw_prob = n2.predict_image(image)  # Ensure this expects PIL.Image
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference error: {str(e)}")

    return {
        'age': predicted_age,
        "url": request.url,
        "score": round(nsfw_prob, 4),
        "sfw_probability": round(1 - nsfw_prob, 4),
        "nsfw_probability": round(nsfw_prob, 4),
        "is_nsfw": nsfw_prob > threshold,
        "classification": classify_score(nsfw_prob)
    }

# -------------------------------------
# 2. Classify Uploaded Video File
# -------------------------------------
@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to save uploaded video.")

    try:
        elapsed_seconds, nsfw_probs = n2.predict_video_frames(tmp_path, frame_interval=10)
    except Exception as e:
        os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=f"Video inference error: {str(e)}")

    os.remove(tmp_path)

    results = [
        {
            "time": round(t, 2),
            "score": round(prob, 4),
            "sfw_probability": round(1 - prob, 4),
            "nsfw_probability": round(prob, 4),
            "is_nsfw": prob > NSFW_THRESHOLD,
            "classification": classify_score(prob)
        }
        for t, prob in zip(elapsed_seconds, nsfw_probs)
    ]

    return {
        "filename": file.filename,
        "frame_results": results
    }

# -------------------------------------
# 3. Classify Batch Image URLs
# -------------------------------------
class ImageBatchRequest(BaseModel):
    urls: List[HttpUrl]

@app.post("/classify-images")
async def classify_images_by_url(request: ImageBatchRequest):
    images = []

    for url in request.urls:
        try:
            res = requests.get(url, timeout=10)
            res.raise_for_status()
            img = Image.open(io.BytesIO(res.content)).convert("RGB")
            images.append(img)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load image: {url} — {str(e)}")

    try:
        scores = n2.predict_images(images)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    results = []
    for url, score in zip(request.urls, scores):
        results.append({
            "url": url,
            "score": round(score, 4),
            "sfw_probability": round(1 - score, 4),
            "nsfw_probability": round(score, 4),
            "is_nsfw": score > NSFW_THRESHOLD,
            "classification": classify_score(score)
        })

    return results
