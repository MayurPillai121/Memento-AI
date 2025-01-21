FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libx11-6 \
    build-essential \
    git \
    wget \
    libpng-dev \
    libjpeg-dev \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python packages with optimizations
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

# Create a non-root user
RUN useradd -m -s /bin/bash appuser

# Install system dependencies first
RUN apt-get update && apt-get install -y \
    cmake \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages in correct order
RUN pip install --upgrade pip && \
    pip install cmake>=3.25.0 && \
    pip install numpy>=1.24.0 && \
    pip install dlib==19.24.1 && \
    pip install -r requirements.txt

# Change ownership of app directory
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p uploads weights

# Download YOLOv5 weights (specific version)
RUN wget -q https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt -P weights/

# Cleanup to reduce image size
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    rm -rf ~/.cache/pip/*

# Set environment variables for production
ENV FLASK_ENV=production \
    FLASK_APP=app.py

# Use gunicorn as the production server
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
