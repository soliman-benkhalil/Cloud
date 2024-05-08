# Base Image Selection
FROM python:3.9-slim-buster  

# Set working directory
WORKDIR / C:\dockerize\Project\project 2
# sets the working directory for any subsequent instructions in the Dockerfile

# Install system dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    apt-get clean

# Copy only the requirements file first to leverage Docker's caching mechanism
COPY requirements.txt .

# Install dependencies (Python packages)
RUN pip install --no-cache-dir -r requirements.txt


# Copy the rest of the application code
COPY . .


# Set the command to run the application
CMD ["python", "cloud.py"]

