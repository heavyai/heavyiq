# Use the official Python image as the base image
FROM ubuntu:latest

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y python3.10 python3.10-venv git libsndfile1 build-essential python3.10-dev python3.10-distutils && \
    rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment
RUN python3 -m venv venv
ENV PATH="/app/venv/bin:$PATH"

# Copy requirements
COPY requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

# Copy the entrypoint script into the container
COPY entrypoint.sh /app/entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /app/entrypoint.sh

ENV PORT=8001
# Install the whisper-cpp-python[server] package
#RUN pip install git+https://github.com/afafw/whisper-cpp-python.git && pip install uvicorn>=0.21.1
EXPOSE 8001

# Command to run on container start
CMD ["/app/entrypoint.sh"]
