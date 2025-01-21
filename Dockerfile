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
    git \
    wget \
    libpng-dev \
    libjpeg-dev \
    libopenblas-dev \
    liblapack-dev \
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

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    python3-dev \
    libx11-dev \
    libatlas-base-dev \
    libgtk-3-dev \
    libboost-python-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for dlib
ENV CFLAGS="-O2"
ENV CXXFLAGS="-O2"

# Install base Python packages
RUN pip install --upgrade pip wheel setuptools numpy

# Install dlib from source with optimizations
RUN git clone --depth 1 https://github.com/davisking/dlib.git && \
    cd dlib && \
    python setup.py install --no USE_AVX_INSTRUCTIONS && \
    cd .. && \
    rm -rf dlib

# Install remaining Python packages
RUN pip install -r requirements.txt

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
