# Use the latest Ubuntu image as a parent image
FROM ubuntu:latest

# Set the working directory to /app
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y python3.10 python3.10-venv build-essential python3.10-dev python3.10-distutils && \
    rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment
RUN python3 -m venv venv
ENV PATH="/app/venv/bin:$PATH"

# Copy the requirements.txt file into the container at /app
COPY requirements.txt ./
COPY requirements-dev.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-dev.txt

# COPY config.toml
COPY ./config.test.toml ./config.toml

# Copy the current directory contents into the container at /app
COPY ./heavyiq /app/heavyiq

# copy tests
COPY ./tests /app/tests

#Define the command to run on container start
CMD ["pytest", "tests", "--disable-warnings", "-rs", "--config-path=./config.toml"]
