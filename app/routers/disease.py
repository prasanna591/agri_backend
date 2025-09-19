# filename: disease_detector.py

import os
import google.generativeai as genai
from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image
import io
import json
from dotenv import load_dotenv

# --- Load Environment Variables (can be done in main.py, but safe to have here too) ---
load_dotenv()

# --- Create a Router instead of a full FastAPI app ---
router = APIRouter(
    prefix="/disease",
    tags=["Disease Detector"],
)

# --- Gemini API Configuration ---
API_KEY = os.getenv("GOOGLE_API_KEY") # Ensure this is the correct variable name from your .env file

if not API_KEY:
    print("ERROR: GOOGLE_API_KEY not found. Disease detector will not work.")
    model = None
else:
    try:
        genai.configure(api_key=API_KEY)
        # Switched to the latest supported model for vision tasks
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        print("Disease detector model loaded successfully.")
    except Exception as e:
        print(f"Error configuring Gemini API for disease detector: {e}")
        model = None

# --- Prompt Configuration ---
system_prompt = """
You are an expert botanist and plant pathologist. Your task is to analyze an image of a plant leaf and identify any diseases.

Your response must be in a JSON format with the following structure:
{
  "disease_name": "Name of the disease or 'Healthy'",
  "precautions": [
    "Precaution 1",
    "Precaution 2"
  ],
  "remedies": [
    "Remedy 1",
    "Remedy 2"
  ],
  "medicines": [
    {
      "name": "Medicine Name 1",
      "mixing_ratio": "Mixing ratio for Medicine 1 (e.g., '10ml per 1 liter of water')"
    }
  ]
}

If the plant in the image appears healthy, set the "disease_name" to "Healthy" and provide general plant care tips in the other fields. If the image is not of a plant or the disease cannot be identified, return an appropriate message within the JSON structure, perhaps setting the disease_name to "Identification Failed".
"""

# --- API Endpoint Definition ---
@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receives an image file, sends it to Gemini for analysis,
    and returns the structured disease information.
    """
    if not model:
        raise HTTPException(status_code=503, detail="AI Model is not available or configured correctly.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # --- Gemini API Call ---
        response = model.generate_content([system_prompt, image])
        
        # Extract and clean the JSON text from the response
        json_text = response.text.strip()
        if json_text.startswith("```json"):
            json_text = json_text[7:-3].strip() # Remove markdown code block

        result = json.loads(json_text)
        return result

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing the image: {e}")