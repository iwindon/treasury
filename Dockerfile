FROM python:3.11-slim

# Install tesseract and required system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
	tesseract-ocr \
	libtesseract-dev \
	libleptonica-dev \
	build-essential \
	pkg-config \
	# OpenCV runtime dependencies
	libglib2.0-0 \
	libsm6 \
	libxrender1 \
	libxext6 \
	libgl1 \
	# PaddlePaddle / numerical runtime deps
	libgomp1 \
	libopenblas-dev \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
# Install Python requirements (EasyOCR will be installed from requirements)
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
