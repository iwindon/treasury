FROM python:3.11-slim

# Install minimal runtime deps for OpenCV and numeric libraries used by EasyOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
	libglib2.0-0 \
	libsm6 \
	libxrender1 \
	libxext6 \
	libgomp1 \
	libopenblas0 \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
# Install Python requirements (EasyOCR and dependencies)
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
