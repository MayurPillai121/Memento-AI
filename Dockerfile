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
    build-essential \
    cmake \
    pkg-config \
    python3-dev \
    libx11-dev \
    libatlas-base-dev \
    libgtk-3-dev \
    libboost-python-dev \
    gfortran \
    python3-setuptools \
    python3-pip \
    python3-wheel \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Create a non-root user
RUN useradd -m -s /bin/bash appuser

# Set environment variables for dlib and Python
ENV CFLAGS="-O2" \
    CXXFLAGS="-O2" \
    USE_AVX_INSTRUCTIONS=0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

# Install base Python packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel numpy cmake

# Install dlib from source with optimizations
RUN git clone --depth 1 https://github.com/davisking/dlib.git && \
    cd dlib && \
    python setup.py install && \
    cd .. && \
    rm -rf dlib

# Install all other requirements
RUN pip install --no-cache-dir -r requirements.txt

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

# Set environment variables for production
ENV FLASK_ENV=production \
    FLASK_APP=app.py

# Use gunicorn as the production server
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
