FROM python:3.10-slim

# FFmpeg & System Dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Shell format used so $PORT is read dynamically from Render
CMD sh -c "gunicorn -w 1 -b 0.0.0.0:${PORT:-5000} app:app"