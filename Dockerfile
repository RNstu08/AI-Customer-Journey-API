# Use an official lightweight Python image as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Install build essentials for packages that need compilation
# This adds necessary tools like gcc (cc), make, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    # Add any other specific libs if necessary later, e.g., libgfortran5 for some numpy/scipy builds
    # libpq-dev for psycopg2-binary
    # libsndfile1 for torchaudio
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker's layer caching
COPY ./requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# Expose the port the app runs on
EXPOSE 8080

# Define the command to run your app using uvicorn
# We use --host 0.0.0.0 to make it accessible from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]