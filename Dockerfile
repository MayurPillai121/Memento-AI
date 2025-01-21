# Use Python 3.10 slim base image
FROM python:3.10-slim

WORKDIR /app

# Set environment variables
ENV CFLAGS="-O2" \
    CXXFLAGS="-O2" \
    USE_AVX_INSTRUCTIONS=0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    DEBIAN_FRONTEND=noninteractive \
    FLASK_ENV=production \
    FLASK_APP=app.py

# Install system dependencies with error handling
RUN set -ex && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        # Basic utilities
        wget \
        git \
        # X11 and graphics
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libx11-6 \
        libx11-dev \
        libgtk-3-dev \
        # Development tools
        build-essential \
        cmake \
        pkg-config \
        python3-dev \
        python3-setuptools \
        python3-pip \
        python3-wheel \
        # Scientific computing
        libopenblas-dev \
        liblapack-dev \
        gfortran \
        libatlas-base-dev \
        # Image processing
        libpng-dev \
        libjpeg-dev \
        libboost-python-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -s /bin/bash appuser

# Copy requirements first
COPY requirements.txt .

# Install Python packages in stages with error handling and logging
RUN set -ex && \
    # Base packages
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy cmake

# Install PyTorch and TensorFlow separately
RUN set -ex && \
    pip install --no-cache-dir tensorflow-cpu>=2.4.1 && \
    pip install --no-cache-dir torch==2.1.2+cpu torchvision==0.16.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Install dlib with verbose output
RUN set -ex && \
    git clone --depth 1 https://github.com/davisking/dlib.git && \
    cd dlib && \
    python setup.py install --verbose && \
    cd .. && \
    rm -rf dlib

# Install remaining packages in groups with error handling
RUN set -ex && \
    echo "Installing Web Framework packages..." && \
    pip install --no-cache-dir Flask gunicorn && \
    echo "Installing Computer Vision packages..." && \
    pip install --no-cache-dir opencv-python-headless Pillow mediapipe-silicon face-recognition-models imutils easyocr deepface && \
    echo "Installing Data Science packages..." && \
    pip install --no-cache-dir pandas matplotlib seaborn scipy scikit-learn scikit-image && \
    echo "Installing AI/NLP packages..." && \
    pip install --no-cache-dir transformers openai && \
    echo "Installing Utility packages..." && \
    pip install --no-cache-dir PyYAML tqdm protobuf python-dotenv tensorflow-compression tensorflow-model-optimization && \
    echo "Installing YOLOv5..." && \
    pip install --no-cache-dir yolov5==6.1.0

# Create necessary directories
RUN mkdir -p uploads weights

# Download YOLOv5 weights
RUN wget -q https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt -P weights/

# Copy application files
COPY . .

# Set permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Add verbose pip installation logging
RUN pip list

# Start the application
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
