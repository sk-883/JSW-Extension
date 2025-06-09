FROM python:3.10-slim

# Don’t generate .pyc files and ensure logs go straight to console
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port Uvicorn will listen on
EXPOSE 8080

# Expose port and launch via Uvicorn
# CMD ["gunicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:8080", "--workers", "2"]
