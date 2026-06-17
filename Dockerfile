# base image
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# metadata
LABEL project="comic-search"
LABEL authors="charles,dayas,rusty"

# system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# working directory
WORKDIR /app

# python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# copy project code
COPY *.py .

# volumes
VOLUME ["/app/datasets", "/app/dataset"]

# default command
CMD ["python3", "main.py"]