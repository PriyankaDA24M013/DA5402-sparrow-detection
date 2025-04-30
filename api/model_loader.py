"""Load the trained Faster R-CNN model for house sparrow detection."""

import os
import torch
from pathlib import Path
from typing import Optional, Dict, Any

# Import function to create model
from src.model import faster_rcnn_mob_model_for_n_classes


class ModelLoader:
    """Singleton class to load and manage the detection model."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._model = None
            cls._instance._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return cls._instance
    
    def load_model(self, model_path: str) -> None:
        """Load the model from a checkpoint file.
        
        Args:
            model_path: Path to the model checkpoint file (.pth)
        """
        # Create model with 2 classes (background and house sparrow)
        self._model = faster_rcnn_mob_model_for_n_classes(num_classes=2)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self._device)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move model to device and set to evaluation mode
        self._model.to(self._device)
        self._model.eval()
        
        print(f"Model loaded successfully from {model_path}")
        print(f"Using device: {self._device}")
    
    @property
    def model(self):
        """Get the loaded model."""
        if self._model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        return self._model
        
    @property
    def device(self):
        """Get the device."""
        return self._device