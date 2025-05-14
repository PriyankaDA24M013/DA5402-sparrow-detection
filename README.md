


#  MLOps Object(Sparrow) Detection Project 


This project is an end-to-end MLOps system for training, serving, and monitoring an object detection model using:
- **DVC** for data version control
- **Docker** for orchestration
- **Prometheus** for monitoring
- **Flask** for serving the model via API
- **Streamlit** for frontend UI
- **Grafana** for interactive visualization
- **Python scripts** for model training and utility functions

---

##  Project Structure

```
mlops-object-detection/
├── data/                    # Datasets managed via DVC
├── models/                  # Trained models tracked with DVC
|── mlflow/                  # Tracking model traning with mlflow
├── frontend/
    ├── wrong_predictions    # all the wrongly classified images get stored here 
    ├── app.py               # contains app.py for UI 
├── grafana/                 # An interactive dashboard using prometheus
├── src/
    ├── data_prepare.py      # Performing EDA 
    ├── model.py             # contain Faster RCNN model       
    ├── train.py             # Model training script
├── api
    ├── model_loader.py         # loads best model from mlflow
    ├── detection.py         # Detection of uploaded image
    ├── main.py              # API server exposing inference endpoints
├── docker-compose.yml       # Orchestrates services
├── dvc.yaml                 # DVC pipeline config
├── params.yaml              # Training hyperparameters
├── prometheus.yml           # Prometheus config file
├── requirements.txt         # Training dependencies
├── api_requirements.txt     # API dependencies
├── app_requirements.txt     # UI dependencies (if present)
├── utils.py                 # Shared utility functions
└── README.md                # Project summary
```

---
![Alt text](E:\mlops_project\DA5402-sparrow-detection\mlops_flow.png)

##  Setup Instructions

### 1️ Install Dependencies

**Training dependencies:**
```bash
pip install -r requirements.txt
```

**API dependencies:**
```bash
pip install -r api_requirements.txt
```

**UI dependencies :**
```bash
pip install -r app_requirements.txt
```

---

##  Data Version Control (DVC)

**Pull existing data:**
```bash
dvc pull
```

**Add new data:**
```bash
dvc add data/your_dataset
git add data/.gitignore data/your_dataset.dvc
git commit -m "Added new dataset"
dvc push
```

---

##  Model Training

**Modify hyperparameters:**
Edit `params.yaml`

**Train the model:**
```bash
python train.py
```

**Training logs:**  
Check `train.log` for progress and issues.

---

##  Serving the Model (API)

**Run API locally:**
```bash
python api.py
```

**Or via Docker Compose:**
```bash
docker-compose up --build
```

API runs on: `http://localhost:5000/`

---
##  Webpage for the application 
UI runs on :[http://localhost:8501](http://localhost:8501)
##  Monitoring (Prometheus)

Prometheus config is in `prometheus.yml`.

**Prometheus Dashboard:**  
[http://localhost:9090](http://localhost:9090)

View metrics and system health.

---

##  API Endpoints

| Endpoint   | Method | Description                    |
|------------|:---------|:--------------------------------|
| `/predict`  | POST    | Upload an image and get detections |
| `/metrics`  | GET     | Expose Prometheus metrics         |

---

##  Adding a New Model

1. Place dataset under `data/`
2. Track with DVC:
    ```bash
    dvc add data/your_dataset
    git add data/.gitignore data/your_dataset.dvc
    git commit -m "new dataset"
    dvc push
    ```
3. Update `params.yaml`
4. Train with:
    ```bash
    python train.py
    ```
5. Replace old model in `models/` if needed
6. Update `api.py` to load the new model
7. Restart API:
    ```bash
    docker-compose restart api
    ```

---

##  Docker Compose

**Run everything together:**
```bash
docker-compose up --build
```

**Stop services:**
```bash
docker-compose down
```

---

##  Notes

- Ensure DVC remote is configured for data sharing
- Prometheus targets API metrics for system health
- Check `train.log` for model training details
- `utils.py` has helper utilities used in both training and inference





