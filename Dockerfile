# === Stage 1: Build the Next.js Frontend ===
# Use the official Node.js image as a builder
FROM node:18-alpine AS builder

# Set the working directory for the frontend build
WORKDIR /app

# Copy package.json and package-lock.json to leverage Docker layer caching
COPY frontend/package*.json ./

# Install frontend dependencies
RUN npm install

# Copy the rest of the frontend source code
COPY ./frontend .

# Build the Next.js application for production
RUN npm run build


# === Stage 2: Setup the Python Backend and Final Image ===
# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install Python dependencies
COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the built Next.js app from the 'builder' stage
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/package.json .

# Copy the Python API code
COPY frontend/api ./api

# Expose the port Next.js runs on
EXPOSE 3000

# The command to start the Next.js production server
# This server will also handle your /api routes automatically
CMD ["npm", "run", "start"]