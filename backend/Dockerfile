# Use official Python slim image
FROM python:3.10-slim

WORKDIR /app

# Copy requirements if you have one (create it if not)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy all your app files
COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
