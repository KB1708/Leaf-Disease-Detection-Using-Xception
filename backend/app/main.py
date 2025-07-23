from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .services import prediction
import logging

# Initialize the FastAPI application
app = FastAPI(title="Leaf Disease Detection API")

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CORS Middleware ---
# Allows the frontend (on a different URL) to communicate with this backend.
# In production, restrict this to your actual frontend domain for security.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# --- API Endpoints ---
@app.get("/", tags=["Health Check"])
async def read_root():
    """A simple health check endpoint to confirm the API is running."""
    return {"message": "Welcome to the Leaf Disease Detection API!"}

@app.post("/predict", tags=["Prediction"])
async def predict_image(file: UploadFile = File(...)):
    """
    This endpoint receives an image, uses the ML model to predict the disease,
    and returns the predicted class and confidence score.
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image.")
    
    try:
        image_bytes = await file.read()
        logger.info("Image received, making prediction...")
        result = prediction.get_prediction(image_bytes)
        logger.info(f"Prediction successful: {result}")
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")