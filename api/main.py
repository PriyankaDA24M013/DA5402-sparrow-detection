"""FastAPI application for house sparrow detection."""

import os
import io
import base64
from pathlib import Path
from typing import Dict, Optional, List, Any

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from api.model_loader import ModelLoader
from api.detection import process_image, save_incorrect_image


# Create the FastAPI app
app = FastAPI(title="House Sparrow Detection API")

# Add CORS middleware to allow requests from Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define response models
class DetectionResponse(BaseModel):
    success: bool
    message: str
    detections: Optional[List[Dict[str, Any]]] = None
    annotated_image: Optional[str] = None  # Base64 encoded image


@app.on_event("startup")
async def startup_event():
    """Load the model when the application starts."""
    model_path = os.environ.get("MODEL_PATH", "outputs/models/best_model.pth")
    
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}")
        print("Please set the MODEL_PATH environment variable to your .pth file path")
        return
    
    try:
        model_loader = ModelLoader()
        model_loader.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")


@app.post("/detect/", response_model=DetectionResponse)
async def detect_sparrows(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5)
):
    """Detect house sparrows in the uploaded image.
    
    Args:
        file: Image file to process
        confidence_threshold: Minimum confidence score (0.0-1.0)
        
    Returns:
        JSON response with detection results and annotated image
    """
    try:
        # Read image file
        image_bytes = await file.read()
        
        # Process image
        annotated_image, detections = process_image(
            io.BytesIO(image_bytes), 
            confidence_threshold
        )
        
        # Convert the annotated image to base64 for response
        buffered = io.BytesIO()
        annotated_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Return response
        return DetectionResponse(
            success=True,
            message=f"Detection completed. Found {len(detections)} sparrows.",
            detections=detections,
            annotated_image=img_str
        )
        
    except Exception as e:
        return DetectionResponse(
            success=False,
            message=f"Error processing image: {str(e)}"
        )


@app.post("/feedback/")
async def submit_feedback(
    file: UploadFile = File(...),
    is_correct: bool = Form(...)
):
    """Submit feedback on detection results.
    
    Args:
        file: Image file that was processed
        is_correct: Whether the detection was correct
        
    Returns:
        JSON response confirming feedback receipt
    """
    try:
        if not is_correct:
            # Save to wrong_classified folder
            image_bytes = await file.read()
            output_dir = Path("data/wrong_classified")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate a unique filename
            import time
            filename = f"wrong_{int(time.time())}_{file.filename}"
            output_path = output_dir / filename
            
            # Save the image
            save_incorrect_image(io.BytesIO(image_bytes), str(output_path))
            
            return {"success": True, "message": f"Image saved to wrong_classified folder as {filename}"}
        else:
            return {"success": True, "message": "Thank you for the positive feedback!"}
            
    except Exception as e:
        return {"success": False, "message": f"Error processing feedback: {str(e)}"}


# Health check endpoint
@app.get("/health/")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)