# Dockerfile for YOLOv11 Training on Vertex AI
# Based on Vertex AI's PyTorch GPU training image
FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest

# Set working directory
WORKDIR /app

COPY requirements.txt /app/requirements.txt

# Install Python dependencies
# Upgrade pip first to avoid dependency conflicts
# Using --no-cache-dir to reduce image size
RUN pip install --upgrade pip --no-cache-dir && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy training script
COPY train_yolo_vertex.py /app/train_yolo_vertex.py

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Default command (can be overridden in config.yaml)
CMD ["python", "/app/train_yolo_vertex.py"]
