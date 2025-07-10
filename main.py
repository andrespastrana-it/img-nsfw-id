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
        response = requests.get(request.url, timeout=8000)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        np_image = np.array(image)
        np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load or decode image: {str(e)}")

    predicted_age = "Unknown"
    gender = "Unknown"
    face_detected = False
    faces_count = 0

    try:
        # Resize image if too large for better detection
        height, width = np_image.shape[:2]
        scale_factor = 1.0
        if max(height, width) > 1024:
            if height > width:
                new_height = 1024
                new_width = int(width * (1024 / height))
            else:
                new_width = 1024
                new_height = int(height * (1024 / width))
            
            resized_image = cv2.resize(np_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            scale_factor = max(height, width) / 1024
        else:
            resized_image = np_image

        # Convert to grayscale
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        
        # Try multiple detection methods
        faces = []
        detection_methods = [
            # Method 1: Default parameters
            {'image': gray, 'params': {'scaleFactor': 1.1, 'minNeighbors': 5}},
            
            # Method 2: More sensitive parameters
            {'image': cv2.equalizeHist(gray), 'params': {'scaleFactor': 1.05, 'minNeighbors': 3, 'minSize': (20, 20)}},
            
            # Method 3: Very sensitive
            {'image': cv2.GaussianBlur(cv2.equalizeHist(gray), (3, 3), 0), 'params': {'scaleFactor': 1.03, 'minNeighbors': 2, 'minSize': (15, 15)}},
            
            # Method 4: Large faces only
            {'image': gray, 'params': {'scaleFactor': 1.2, 'minNeighbors': 6, 'minSize': (50, 50)}},
            
            # Method 5: With scale image flag
            {'image': cv2.equalizeHist(gray), 'params': {'scaleFactor': 1.05, 'minNeighbors': 3, 'flags': cv2.CASCADE_SCALE_IMAGE}},
        ]
        
        for i, method in enumerate(detection_methods):
            try:
                faces = face_cascade.detectMultiScale(method['image'], **method['params'])
                if len(faces) > 0:
                    print(f"Faces detected using method {i+1}: {len(faces)} faces")
                    break
            except Exception as e:
                print(f"Detection method {i+1} failed: {str(e)}")
                continue
        
        faces_count = len(faces)
        print(f"Total detected faces: {faces_count}")
        
        if faces_count > 0:
            face_detected = True
            
            # Scale face coordinates back if image was resized
            if scale_factor != 1.0:
                faces = [(int(x * scale_factor), int(y * scale_factor), 
                         int(w * scale_factor), int(h * scale_factor)) for (x, y, w, h) in faces]
            
            # Process the largest face (most likely to be the main subject)
            largest_face = max(faces, key=lambda face: face[2] * face[3])
            x, y, w, h = largest_face
            
            # Add padding around the face for better detection
            padding = int(0.1 * min(w, h))
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(np_image.shape[1] - x, w + 2 * padding)
            h = min(np_image.shape[0] - y, h + 2 * padding)
            
            face_img = np_image[y:y+h, x:x+w]
            
            # Validate face region
            if face_img.shape[0] >= 30 and face_img.shape[1] >= 30:
                mean_intensity = np.mean(face_img)
                if 20 <= mean_intensity <= 235:  # Not too dark or bright
                    try:
                        blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                        
                        # Gender prediction
                        gender_net.setInput(blob)
                        gender_preds = gender_net.forward()
                        gender = gender_list[gender_preds[0].argmax()]
                        
                        # Age prediction
                        age_net.setInput(blob)
                        age_preds = age_net.forward()
                        predicted_age = age_list[age_preds[0].argmax()]
                        
                        print(f"Predicted - Age: {predicted_age}, Gender: {gender}")
                    except Exception as e:
                        print(f"Age/Gender prediction failed: {str(e)}")
                else:
                    print(f"Face region lighting issue - mean intensity: {mean_intensity}")
            else:
                print(f"Face region too small: {face_img.shape}")
        else:
            print("No faces detected in image")

        # NSFW check (independent of face detection)
        nsfw_prob = n2.predict_image(image)
        
    except Exception as e:
        print(f"Model inference error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model inference error: {str(e)}")

    return {
        'age': predicted_age,
        'gender': gender,
        'face_detected': face_detected,
        'faces_count': faces_count,
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
