FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY ./api ./api
COPY ./utils.py ./utils.py
COPY ./src ./src

ENV MODEL_PATH=outputs/models/best_model.pth

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
