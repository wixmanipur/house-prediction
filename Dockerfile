# Use an official Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required for OpenCV and YOLO
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Verify all required Python imports at build time (Fails early if an issue exists)
RUN python -c "import fastapi, uvicorn, numpy, cv2, PIL, ultralytics; print('All dependencies successfully imported!')"

# Copy application files
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Run the FastAPI app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
