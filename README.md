# Label OCR Prototype

This repository contains a small prototype FastAPI application that accepts label images and performs OCR + simple extraction of label fields (brand, class/type, alcohol content, net contents, and a government warning presence check).

Quick start (local, requires Docker):

1. Build the image:
   docker build -t label-ocr-prototype .

2. Run the container:
   docker run -p 8000:8000 label-ocr-prototype

3. Upload an image:
   POST to http://localhost:8000/upload with form-data key `image` (file)

Notes & assumptions:
- This prototype uses local EasyOCR (models baked into the image) and heuristic regexes to extract fields.
- Image preprocessing is basic; results depend on image quality.
- Government warning detection is a substring match against a canonical paragraph and may need refinement.

Files of interest:
- app.py - FastAPI application
- ocr_extractor.py - OCR and extraction logic
- Dockerfile / requirements.txt - for building the prototype

Deploy to Azure (Container Apps):
   az login
   az extension add --name containerapp
   ./deploy-azure.ps1
The script creates a resource group and ACR, builds the image remotely, deploys to Azure Container Apps (2 vCPU / 4 GiB, min 1 replica to avoid OCR cold starts), and prints the public URL.
