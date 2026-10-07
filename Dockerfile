FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
	libglib2.0-0 \
	libsm6 \
	libxrender1 \
	libxext6 \
	libgomp1 \
	libopenblas0 \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch keeps the image small enough for fast Azure cold starts
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Bake EasyOCR models into the image so the app works offline and starts fast
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

COPY . /app

ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]