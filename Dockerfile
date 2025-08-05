# === Stage 1: Build the Next.js Frontend ===
# This stage builds your frontend into an optimized package

FROM node:18-alpine AS builder

# Set the working directory inside the build container
WORKDIR /app

# Copy the frontend code into the container
# The '.' after 'frontend/' is important
COPY frontend/ .

# Install dependencies and build the Next.js app for production
RUN npm install
RUN npm run build


# === Stage 2: Create the Final Production Image ===
# This stage takes the built frontend and adds the Python backend

FROM python:3.11-slim

# Set the working directory in the final container
WORKDIR /app

# Set the PORT environment variable that Vercel and Next.js use
ENV PORT=3000

# Copy the Python requirements file from your repo
COPY frontend/requirements.txt .
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the essential built files from the 'builder' stage
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/package.json .
COPY --from=builder /app/next.config.js .

# Copy your Python API code into the final image
COPY frontend/api ./api

# Expose the port the server will run on
EXPOSE 3000

# The command to start the Next.js production server.
# This server will handle both the frontend pages and the /api routes.
CMD ["npm", "run", "start"]