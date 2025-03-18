# Use Python 3.9 slim as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Expose FastAPI port
EXPOSE 8000

# Keeps Python from generating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create a non-root user for security
RUN adduser --uid 5678 --disabled-password --gecos "" appuser && \
   chown -R appuser /app
USER appuser

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
