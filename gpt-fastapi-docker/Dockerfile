FROM python:3.9-slim

# Set working directory in the container
WORKDIR /app

# Copy requirements.txt to container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
