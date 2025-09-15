# app.py
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn

from utils import load_model_and_classes, predict_severity_from_image, fetch_weather, compute_weather_risk, compute_recommended_dose, natural_language_output
from pydantic import BaseModel

# Provide your class names (in the same order your model was trained)
CLASS_NAMES = ["healthy", "mild_disease", "moderate_disease", "severe_disease"]

# Pydantic models for request bodies
class WeatherData(BaseModel):
    temp: float
    humidity: float
    rainfall: float

class PesticideData(BaseModel):
    crop: str
    disease_severity: int
    humidity: float
    rainfall: float

# Load model once (with error handling for missing model)
try:
    model, _ = load_model_and_classes("disease_model.h5", class_names=CLASS_NAMES)
    print("✅ Disease model loaded successfully")
except FileNotFoundError:
    print("⚠️  Disease model not found. Please run 'python model_train.py' to train the model first.")
    print("   Using dummy model for testing purposes...")
    model = None

app = FastAPI(title="AgriDose API")

@app.get("/")
async def root():
    return {"message": "AgriDose API is running!", "docs": "/docs"}

@app.post("/predict_disease/")
async def predict_disease(file: UploadFile = File(...)):
    """Predict disease severity from uploaded image"""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": "Invalid image file", "details": str(e)})

    # Predict disease severity
    if model is not None:
        severity_score, probs = predict_severity_from_image(image, model, CLASS_NAMES)
    else:
        # Dummy prediction for testing when model is not available
        severity_score = 0.3  # Mild disease for testing
        probs = [0.1, 0.7, 0.15, 0.05]  # Dummy probabilities
        print("⚠️  Using dummy prediction - model not loaded")

    # Get the predicted class and confidence
    predicted_class = CLASS_NAMES[np.argmax(probs)] if model is not None else "mild_disease"
    confidence = max(probs) if model is not None else 0.7
    
    # Create a mock recommendation for natural language output
    mock_recommendation = {
        "recommended_dose_ml_per_ha": 2000,
        "interval_days": 7,
        "legal_max_ml_per_ha": 3000
    }
    
    # Generate natural language output
    natural_language = natural_language_output(predicted_class, confidence, mock_recommendation)
    
    return {
        "severity_score": round(float(severity_score), 3),
        "probabilities": probs,
        "class_names": CLASS_NAMES,
        "prediction": predicted_class,
        "confidence": round(float(confidence), 3),
        "natural_language": natural_language
    }

@app.post("/recommend")
async def recommend(
    file: UploadFile = File(...),
    lat: float = Form(...),
    lon: float = Form(...),
    crop_stage: str = Form(...),
    base_dose_ml_per_ha: float = Form(...),
    pesticide_name: str = Form("example_pesticide")
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": "Invalid image file", "details": str(e)})

    # 1) severity from image
    if model is not None:
        severity_score, probs = predict_severity_from_image(image, model, CLASS_NAMES)
    else:
        # Dummy prediction for testing when model is not available
        severity_score = 0.3  # Mild disease for testing
        probs = [0.1, 0.7, 0.15, 0.05]  # Dummy probabilities
        print("⚠️  Using dummy prediction - model not loaded")

    # 2) fetch weather and compute risk
    try:
        weather_json = fetch_weather(lat, lon)
        weather_risk = compute_weather_risk(weather_json)
    except Exception as e:
        # default mild risk if API fails
        weather_risk = 0.2

    # 3) compute dose
    rec = compute_recommended_dose(base_dose_ml_per_ha, severity_score, weather_risk, crop_stage, pesticide_name)

    # 4) Generate natural language output
    predicted_class = CLASS_NAMES[np.argmax(probs)] if model is not None else "mild_disease"
    confidence = max(probs) if model is not None else 0.7
    natural_language = natural_language_output(predicted_class, confidence, rec)
    
    # 5) response includes underlying data for transparency
    response = {
        "recommendation": rec,
        "model_probs": probs,
        "class_names": CLASS_NAMES,
        "prediction": predicted_class,
        "confidence": round(float(confidence), 3),
        "natural_language": natural_language
    }
    return response

@app.post("/predict_weather/")
async def predict_weather(weather_data: WeatherData):
    """Predict weather risk based on temperature, humidity, and rainfall"""
    try:
        # Create a mock weather JSON for the existing function
        weather_json = {
            "current": {
                "temp": weather_data.temp,
                "humidity": weather_data.humidity
            },
            "daily": [{
                "rain": weather_data.rainfall
            }]
        }
        
        weather_risk = compute_weather_risk(weather_json)
        
        return {
            "weather_risk": round(float(weather_risk), 3),
            "input_data": {
                "temperature": weather_data.temp,
                "humidity": weather_data.humidity,
                "rainfall": weather_data.rainfall
            },
            "risk_level": "High" if weather_risk > 0.7 else "Medium" if weather_risk > 0.4 else "Low"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Weather prediction failed", "details": str(e)})

@app.post("/recommend_pesticide/")
async def recommend_pesticide(pesticide_data: PesticideData):
    """Recommend pesticide dose based on crop, disease severity, and weather conditions"""
    try:
        # Convert disease severity to severity score (1-5 scale to 0-1 scale)
        severity_score = (pesticide_data.disease_severity - 1) / 4.0
        
        # Create mock weather data
        weather_json = {
            "current": {
                "temp": 25,  # Default temperature
                "humidity": pesticide_data.humidity
            },
            "daily": [{
                "rain": pesticide_data.rainfall
            }]
        }
        
        weather_risk = compute_weather_risk(weather_json)
        
        # Use a base dose based on crop type
        base_doses = {
            "wheat": 2000,
            "rice": 2500,
            "maize": 1800,
            "tomato": 2200,
            "potato": 2000
        }
        
        base_dose = base_doses.get(pesticide_data.crop.lower(), 2000)
        
        # Calculate recommended dose
        recommendation = compute_recommended_dose(
            base_dose, 
            severity_score, 
            weather_risk, 
            "vegetative",  # Default crop stage
            "example_pesticide"
        )
        
        # Generate natural language output
        predicted_class = "moderate_disease" if severity_score > 0.5 else "mild_disease"
        confidence = severity_score
        natural_language = natural_language_output(predicted_class, confidence, recommendation)
        
        return {
            "recommendation": recommendation,
            "input_data": {
                "crop": pesticide_data.crop,
                "disease_severity": pesticide_data.disease_severity,
                "humidity": pesticide_data.humidity,
                "rainfall": pesticide_data.rainfall
            },
            "severity_score": round(float(severity_score), 3),
            "weather_risk": round(float(weather_risk), 3),
            "prediction": predicted_class,
            "confidence": round(float(confidence), 3),
            "natural_language": natural_language
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Pesticide recommendation failed", "details": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
