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

# Use environment variable for port (default 8000)
ENV APP_PORT=5000
EXPOSE ${APP_PORT}

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${APP_PORT}"]
