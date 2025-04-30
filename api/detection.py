"""House sparrow detection functionality."""

import io
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as F
from typing import Dict, List, Tuple, Union, BinaryIO

from api.model_loader import ModelLoader


def process_image(image_bytes: BinaryIO, confidence_threshold: float = 0.5) -> Tuple[Image.Image, List[Dict]]:
    """Process an image and detect house sparrows.
    
    Args:
        image_bytes: Image bytes to process
        confidence_threshold: Minimum confidence score to consider a detection valid
        
    Returns:
        Tuple of (annotated image, detection results)
    """
    # Load model
    model_loader = ModelLoader()
    model = model_loader.model
    device = model_loader.device
    
    # Open and process image
    image = Image.open(image_bytes).convert("RGB")
    img_tensor = F.to_tensor(image).to(device)
    
    # Perform inference
    with torch.no_grad():
        prediction = model([img_tensor])[0]
    
    # Extract results
    boxes = prediction["boxes"].cpu().numpy()
    scores = prediction["scores"].cpu().numpy()
    labels = prediction["labels"].cpu().numpy()
    
    # Filter results by confidence threshold and class label (1 = house sparrow)
    sparrow_mask = (scores >= confidence_threshold) & (labels == 1)
    
    filtered_boxes = boxes[sparrow_mask]
    filtered_scores = scores[sparrow_mask]
    
    # Create annotated image
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    # Try to get a font, use default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw bounding boxes
    detection_results = []
    for box, score in zip(filtered_boxes, filtered_scores):
        # Convert box coordinates to integers
        box = box.astype(int)
        x_min, y_min, x_max, y_max = box
        
        # Draw rectangle
        draw.rectangle([(x_min, y_min), (x_max, y_max)], 
                       outline="red", width=3)
        
        # Draw label with confidence
        label_text = f"Sparrow: {score:.2f}"
        draw.rectangle([(x_min, y_min - 20), (x_min + len(label_text) * 10, y_min)],
                      fill="red")
        draw.text((x_min + 5, y_min - 20), label_text, fill="white", font=font)
        
        # Add to results
        detection_results.append({
            "box": box.tolist(),
            "confidence": float(score)
        })
    
    return annotated_image, detection_results


def save_incorrect_image(image_bytes: BinaryIO, output_path: str) -> None:
    """Save an incorrectly classified image to the wrong_classified folder.
    
    Args:
        image_bytes: Image bytes to save
        output_path: Path to save the image
    """
    # Ensure directory exists
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the image
    image = Image.open(image_bytes)
    image.save(output_path)