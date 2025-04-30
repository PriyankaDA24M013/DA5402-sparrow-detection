"""FastAPI application for house sparrow detection with comprehensive monitoring."""

import os
import io
import base64
import socket
import psutil
import platform
from pathlib import Path
from typing import Dict, Optional, List, Any

from fastapi import FastAPI, File, UploadFile, Form, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from api.model_loader import ModelLoader
from api.detection import process_image, save_incorrect_image

# Enhanced monitoring imports
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import multiprocess, CollectorRegistry
import time

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

# System information
HOSTNAME = socket.gethostname()
SYSTEM_INFO = Info("system_info", "System information")
SYSTEM_INFO.info({
    "hostname": HOSTNAME,
    "platform": platform.platform(),
    "python_version": platform.python_version(),
})

# Request metrics
REQUEST_COUNT = Counter("api_request_count", "Total number of requests", ["method", "endpoint", "status_code"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"])
REQUEST_IN_PROGRESS = Gauge("api_requests_in_progress", "Requests currently being processed", ["endpoint"])
CLIENT_METRICS = Counter("api_client_requests", "Client request metrics", ["client_ip", "user_agent"])

# System resource metrics
CPU_USAGE = Gauge("system_cpu_usage", "CPU usage percentage")
MEMORY_USAGE = Gauge("system_memory_usage_bytes", "Memory usage in bytes")
MEMORY_PERCENT = Gauge("system_memory_usage_percent", "Memory usage percentage")
DISK_IO_READ = Counter("system_disk_read_bytes", "Disk read bytes")
DISK_IO_WRITE = Counter("system_disk_write_bytes", "Disk write bytes")
NETWORK_IO_SENT = Counter("system_network_sent_bytes", "Network bytes sent")
NETWORK_IO_RECV = Counter("system_network_recv_bytes", "Network bytes received")
FILE_HANDLES = Gauge("system_open_file_descriptors", "Open file descriptors")
THREAD_COUNT = Gauge("system_thread_count", "Number of threads")

# Model metrics
MODEL_INFERENCE_TIME = Histogram("model_inference_time_seconds", "Model inference time in seconds")
DETECTION_COUNT = Counter("detection_count", "Number of objects detected")
CONFIDENCE_SCORES = Histogram("detection_confidence_scores", "Confidence scores of detections", buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# System metrics collection function
def collect_system_metrics():
    # CPU metrics
    CPU_USAGE.set(psutil.cpu_percent(interval=None))
    
    # Memory metrics
    memory = psutil.virtual_memory()
    MEMORY_USAGE.set(memory.used)
    MEMORY_PERCENT.set(memory.percent)
    
    # Disk I/O metrics
    disk_io = psutil.disk_io_counters()
    DISK_IO_READ.inc(disk_io.read_bytes)
    DISK_IO_WRITE.inc(disk_io.write_bytes)
    
    # Network I/O metrics
    net_io = psutil.net_io_counters()
    NETWORK_IO_SENT.inc(net_io.bytes_sent)
    NETWORK_IO_RECV.inc(net_io.bytes_recv)
    
    # File handles
    try:
        FILE_HANDLES.set(len(psutil.Process().open_files()))
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        pass
    
    # Thread count
    THREAD_COUNT.set(len(psutil.Process().threads()))

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    # Track in-progress requests
    REQUEST_IN_PROGRESS.labels(endpoint=request.url.path).inc()
    
    # Track client metrics
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    CLIENT_METRICS.labels(client_ip=client_ip, user_agent=user_agent).inc()
    
    # Time request
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Record metrics
    duration = time.time() - start_time
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status_code=response.status_code).inc()
    REQUEST_IN_PROGRESS.labels(endpoint=request.url.path).dec()
    
    # Collect system metrics
    collect_system_metrics()
    
    return response

@app.get("/metrics")
def metrics():
    """Endpoint for exposing Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Define response models
class DetectionResponse(BaseModel):
    success: bool
    message: str
    detections: Optional[List[Dict[str, Any]]] = None
    annotated_image: Optional[str] = None  # Base64 encoded image
    processing_time: Optional[float] = None  # Processing time in seconds

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
    """Detect house sparrows in the uploaded image."""
    try:
        # Start timing the inference
        inference_start = time.time()
        
        # Read image file
        image_bytes = await file.read()
        
        # Process image
        annotated_image, detections = process_image(
            io.BytesIO(image_bytes), 
            confidence_threshold
        )
        
        # Record inference time
        inference_time = time.time() - inference_start
        MODEL_INFERENCE_TIME.observe(inference_time)
        
        # Record detection metrics
        DETECTION_COUNT.inc(len(detections))
        for detection in detections:
            CONFIDENCE_SCORES.observe(detection['confidence'])
        
        # Convert the annotated image to base64 for response
        buffered = io.BytesIO()
        annotated_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Return response
        return DetectionResponse(
            success=True,
            message=f"Detection completed. Found {len(detections)} sparrows.",
            detections=detections,
            annotated_image=img_str,
            processing_time=inference_time
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
    """Submit feedback on detection results."""
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

# Extended health check endpoint with system metrics
@app.get("/health/")
async def health_check():
    """Health check endpoint with system metrics."""
    # Collect current system metrics
    collect_system_metrics()
    
    # Get current values
    cpu = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        "status": "healthy",
        "system": {
            "hostname": HOSTNAME,
            "cpu_usage": f"{cpu}%",
            "memory_usage": f"{memory.percent}%",
            "disk_usage": f"{disk.percent}%",
            "uptime": time.time() - psutil.boot_time()
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)