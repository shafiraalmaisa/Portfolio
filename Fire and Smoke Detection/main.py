from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from PIL import Image
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import logging
import tempfile
import shutil
from collections import Counter
import os
import cv2
from tensorflow.keras.models import load_model

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
cnn_model_path = 'D:\TugasAkhir\models\cnn_model244.h5'  
mlp_model_path = 'D:\TugasAkhir\models\mlp_model244.h5'  

cnn_model = load_model(cnn_model_path)
mlp_model = load_model(mlp_model_path)

class_names = ['Smoke', 'Fire', 'Non Fire']

def extract_frames(video_path, interval=1):
    video = cv2.VideoCapture(video_path)
    frames = []
    fps = video.get(cv2.CAP_PROP_FPS)  # Get frames per second (FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps  # Calculate duration in seconds

    # Ensure interval does not exceed video duration
    interval = min(interval, duration)
    # Convert interval to frames
    interval_frames = int(interval * fps) 
    
    count = 0
    success = True
    while success:
        success, frame = video.read()
        if frame is None:
            break
        if count % interval_frames == 0:
            frames.append(frame)
        count += 1
    
    video.release()
    return frames

def preprocess_frame(frame, target_size):
    frame = cv2.resize(frame, target_size)
    frame = img_to_array(frame)
    frame = preprocess_input(frame)
    return frame

def predict_frames(model, frames):
    predictions = []
    for frame in frames:
        processed_frame = preprocess_frame(frame, (244, 244))
        prediction = model.predict(np.expand_dims(processed_frame, axis=0))
        predictions.append(prediction)
    return predictions

def classify_video(video_path, model, interval=1, num_frames=15):
    frames = extract_frames(video_path, interval)
    if len(frames) < num_frames:
        num_frames = len(frames)
    frame_predictions = predict_frames(model, frames[:num_frames])
    
    # Majority voting
    class_ids = [np.argmax(pred) for pred in frame_predictions]
    counter = Counter(class_ids)
    majority_class_id = counter.most_common(1)[0][0]

    # Calculate average probabilities
    average_probabilities = np.mean(np.array([pred.flatten() for pred in frame_predictions]), axis=0)

    return majority_class_id, average_probabilities

@app.post("/predict/")
async def predict(file: UploadFile = File(...), data_type: str = Form(...), model_name: str = Form(...)):
    contents = await file.read()
    
    if data_type == 'image':
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            logging.error(f"Error opening image: {e}")
            return JSONResponse(content={'error': 'Invalid image file'}, status_code=400)
        
        logging.info(f"Original image size: {image.size}")
        
        # Preprocess image
        img = image.resize((244, 244))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        logging.info(f"Transformed image shape: {img_array.shape}")

        try:
            if model_name == 'mlp':
                prediction = mlp_model.predict(img_array)
            elif model_name == 'cnn':
                prediction = cnn_model.predict(img_array)
            else:
                return JSONResponse(content={'error': 'Invalid model name'}, status_code=400)
            
            probabilities = tf.nn.softmax(prediction, axis=1).numpy().flatten()
            logging.info(f"Raw prediction: {prediction}")
            predicted_class = np.argmax(prediction, axis=1).item()

        except Exception as e:
            logging.error(f"Error during model prediction: {e}")
            return JSONResponse(content={'error': 'Model prediction failed'}, status_code=500)

        predicted_class_name = class_names[predicted_class]
        logging.info(f"Predicted class: {predicted_class_name}")

        probabilities_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}

        return {
            "predicted_class": predicted_class_name,
            "probabilities": probabilities_dict
        }
    elif data_type == "video":
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(contents)
                tmp_path = tmp.name

            # Load the appropriate model
            if model_name == 'cnn':
                model = cnn_model
            elif model_name == 'mlp':
                model = mlp_model
            else:
                return {"error": "Invalid model name"}

            # Classify video
            majority_class_id, average_probabilities = classify_video(tmp_path, model)
            predicted_class = class_names[majority_class_id]
            average_prediction = average_probabilities

            # Remove the temporary file
            os.remove(tmp_path)

            return {
                "predicted_class": predicted_class,
                "probabilities": {class_name: float(prob) for class_name, prob in zip(class_names, average_prediction)}
            }
        except Exception as e:
            logging.error(f"Error handling video file: {e}")
            return {"error": "Error processing video file"}
    else:
        return {"error": "Unsupported data type"}
