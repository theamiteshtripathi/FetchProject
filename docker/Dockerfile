FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Create directory for model outputs
RUN mkdir -p outputs

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/outputs/best_model.pt

# Expose ports for API and frontend
EXPOSE 8000 8501

# Create a script to run both services
RUN echo '#!/bin/bash\n\
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 & \n\
streamlit run src/frontend/app.py --server.port 8501 --server.address 0.0.0.0\n\
' > /app/run_services.sh && chmod +x /app/run_services.sh

# Command to run both services
CMD ["/app/run_services.sh"] 