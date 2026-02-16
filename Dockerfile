# Use a slim Python base image
FROM python:3.9-slim

# System deps (optional but helps with some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
# If you have requirements.txt, copy it; otherwise install inline in README section.
COPY docker_req.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY app.py /app/app.py

# Copy model into the container (so it runs standalone)
# If you prefer mounting the model at runtime, you can remove this COPY and use -v in docker run.
COPY distilbert-imdb-best /app/distilbert-imdb-best

# Make the model path configurable
ENV MODEL_DIR=/app/distilbert-imdb-best
ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

# Run the API
CMD ["sh", "-c", "uvicorn app:app --host ${HOST} --port ${PORT}"]
