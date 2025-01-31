# Use the official Python image
FROM python:3.10

# Set environment variables to avoid interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install build tools and other dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    libopenmpi-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the entire project directory into the container
# Copy the directories outside the code folder to the container
COPY ../data /app/data
COPY ../documentation /app/documentation
COPY ../models /app/models
COPY ../solution /app/solution
COPY ../train /app/train
COPY code /app/code

# Install dependencies
RUN pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support
RUN pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu121

# Install prebuilt xformers
RUN pip install xformers==0.0.22

# Install the remaining dependencies
COPY code/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Set the default command to run your pipeline
CMD ["python3", "code/main.py", "--dataset", "provisional"]
