FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libfreetype6-dev \
    libx11-6 \
    fonts-freefont-ttf \
    bash \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app_requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r app_requirements.txt

COPY ./frontend ./frontend

CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]