FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
  git \
  ffmpeg \
  build-essential \
  libgl1-mesa-glx \
  && rm -rf /var/lib/apt/lists/*

# Install Python base packages
RUN pip install --no-cache-dir fastapi uvicorn python-multipart pillow opencv-python-headless

# Install TensorFlow manually (required for OpenNSFW2)
RUN pip install --no-cache-dir tensorflow

# Clone and install OpenNSFW2 from GitHub
WORKDIR /tmp
RUN git clone https://github.com/bhky/opennsfw2.git
WORKDIR /tmp/opennsfw2
RUN pip install .

# Copy FastAPI app
WORKDIR /app
COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
