# 1. Use a lightweight Python image as the base
FROM python:3.11-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Install system dependencies needed by numpy / scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements from the repo into the image
# (We assume requirements.txt is in the project root)
COPY requirements.txt .

# 5. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the FastAPI service code (and model.pkl) into the image
COPY service ./service

# 7. Switch to the service folder
WORKDIR /app/service

# 8. Make logs appear immediately
ENV PYTHONUNBUFFERED=1

# 9. Start the FastAPI app with uvicorn when the container runs
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]