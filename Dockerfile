# Use an official Python 3.11 runtime as a parent image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /var/task

# Copy only the requirements file first to leverage Docker's layer caching
COPY frontend/requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project's code into the container
COPY . .