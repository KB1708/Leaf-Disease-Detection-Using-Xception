# Use an official Python 3.11 runtime as a parent image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /var/task

# Copy the requirements file (it's now in the same directory)
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code from the frontend directory
COPY . .