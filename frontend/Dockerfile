# Use official Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY frontend/requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY frontend/ .
COPY shared/ /app/shared/

# Set Python path to include shared
ENV PYTHONPATH=/app

# Run Streamlit app
CMD ["streamlit", "run", "app.py"]
