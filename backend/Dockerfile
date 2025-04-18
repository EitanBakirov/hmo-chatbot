# Use official Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY backend/requirements.txt .
RUN pip install -r requirements.txt

# Copy all necessary directories
COPY backend/ .
COPY shared/ /app/shared/
COPY scripts/ /app/scripts/
COPY phase2_data/ /app/phase2_data/

# Install beautifulsoup4 for HTML parsing
RUN pip install beautifulsoup4

# Set Python path to include app directory
ENV PYTHONPATH=/app

# Run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
