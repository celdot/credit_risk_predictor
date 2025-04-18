# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR .

# Copy the current directory contents into the container
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    build-essential \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 for the Flask app
EXPOSE 8080

# Define environment variable for Flask
ENV FLASK_APP=app.py

# Run the Flask server
CMD ["python", "app.py", "--port", "8080"]
