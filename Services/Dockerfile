# FROM python:3.10-slim

# # Don’t generate .pyc files and ensure logs go straight to console
# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1

# WORKDIR /app

# # Install dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy application code
# COPY . .

# # Expose the port Uvicorn will listen on
# EXPOSE 8080

# # Expose port and launch via Uvicorn
# # CMD ["gunicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
# CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:8080", "--workers", "2"]


# Use a slim Python base image
FROM python:3.13-slim

# Don’t write .pyc files, and force stdout/stderr to be unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working dir
WORKDIR /app

# Install any OS deps (e.g. build tools)
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy & install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install sentence_transformers
RUN pip install numpy


# Copy application code
COPY . .

# Expose the port your Flask app runs on
EXPOSE 8080

# Run with Gunicorn (2 workers, change as needed)
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:8080", "--workers", "2"]

