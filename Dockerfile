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

# Install base Python packages and dlib first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy==1.24.3 cmake==3.27.7 && \
    git clone --depth 1 https://github.com/davisking/dlib.git && \
    cd dlib && \
    python setup.py install && \
    cd .. && \
    rm -rf dlib

# Install packages in groups to better handle dependencies
RUN pip install --no-cache-dir \
    Flask==3.0.0 \
    gunicorn==21.2.0 \
    Werkzeug==3.0.1 \
    python-dotenv==1.0.0

RUN pip install --no-cache-dir \
    torch==2.1.2+cpu \
    torchvision==0.16.2+cpu \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

RUN pip install --no-cache-dir \
    tensorflow-cpu==2.14.0 \
    tensorflow-hub==0.15.0

RUN pip install --no-cache-dir \
    opencv-python-headless==4.8.1.78 \
    pillow==10.1.0 \
    scipy==1.11.3 \
    pandas==2.1.3 \
    matplotlib==3.8.2

RUN pip install --no-cache-dir \
    scikit-learn==1.3.2 \
    scikit-image==0.22.0 \
    mediapipe==0.10.8 \
    face-recognition-models==0.3.0 \
    imutils==0.5.4

RUN pip install --no-cache-dir \
    transformers==4.35.2 \
    easyocr==1.7.1 \
    deepface==0.0.79 \
    seaborn==0.13.0 \
    PyYAML==6.0.1 \
    tqdm==4.66.1 \
    protobuf==3.20.3 \
    openai==1.3.7

# Change ownership of app directory
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 8080

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
