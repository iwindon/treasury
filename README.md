# Label OCR Prototype

A small FastAPI prototype that accepts label images, runs OCR (EasyOCR), and extracts simple label fields: brand, class/type, alcohol content, net contents, and whether the government warning is present.

## Live deployment (Azure)

The app is deployed to **Azure Container Apps** and is publicly reachable:

**https://label-ocr.agreeabledune-8c207f3e.eastus.azurecontainerapps.io**

| Path | What it does |
|------|--------------|
| `/` | Web UI: upload a label image and see the parsed fields |
| `/docs` | Interactive API docs (Swagger UI) with a "Try it out" upload form |
| `POST /upload` | Upload one image (form key `image`); add `?debug=true` for boxes, per-region OCR and an overlay image |
| `POST /upload/batch` | Upload multiple images (form key `images`) |

The first request after a deploy can take a minute while the OCR models load.

### Try it with curl

```powershell
curl.exe -F "image=@test labels/test_image1.jpg" https://label-ocr.agreeabledune-8c207f3e.eastus.azurecontainerapps.io/upload
```

Example response:

```json
{
  "success": true,
  "data": {
    "raw_text": "...",
    "brand": "IRONWOOD RIDGE SPIRITS",
    "class_type": "Bourbon",
    "alcohol_content": "45% ",
    "net_contents": "750 mL",
    "government_warning_present": true,
    "government_warning_score": 0.966
  }
}
```

## Testing with the sample labels

The [test labels](<./test labels>) folder contains three sample label images you can use to check the application, locally or on the deployed site:

| Image | Product | Expected result |
|-------|---------|-----------------|
| `test_image1.jpg` | Bourbon | brand `IRONWOOD RIDGE SPIRITS`, class `Bourbon`, `45%`, `750 mL`, warning present |
| `test_image2.jpg` | Wine | brand `Silver Creek Cellars`, `13.5%`, `750 mL`, warning present |
| `test_image3.jpg` | Beer | `5.2%`, `12 FL OZ`, warning present |

To test:
1. Open the live URL (or `http://localhost:8000` if running locally).
2. Upload each image through the UI, or via `/docs`, or with the curl command above.
3. Compare the returned fields with the table. OCR output varies slightly, so minor differences (for example in `brand` or spacing) are normal.

Known limitations on these samples: `class_type` only matches a fixed keyword list (bourbon, whiskey, vodka, gin, rum, wine, beer, ...), so it is empty for the wine and beer samples, and `brand` is simply the first line of OCR text, so it can be partial (image 3 returns `RIDGE`).

Add your own images to `test labels` to extend the set. Use `?debug=true` to see why an image parsed poorly.

## Government warning check

The warning is matched fuzzily against the canonical text, ignoring case, spacing and punctuation, so lowercase, uppercase and small OCR errors still match. `government_warning_present` is `true` when `government_warning_score` is at least 0.85 (`WARNING_MATCH_THRESHOLD` in `ocr_extractor.py`). If the warning isn't found on the first pass, the image is re-read in upscaled bands to pick up small print. This is a likelihood check, not a legal compliance check.

## Run locally (requires Docker)

```
docker build -t label-ocr-prototype .
docker run -p 8000:8000 label-ocr-prototype
```

Then open http://localhost:8000 or POST an image to http://localhost:8000/upload with form-data key `image`.

## Deploying to Azure

**First-time setup** provisions the resource group, container registry and Container Apps environment, then deploys:

```
az login
az extension add --name containerapp
./deploy-azure.ps1
```

The script builds the image remotely in ACR (no local Docker needed), deploys to Azure Container Apps (2 vCPU / 4 GiB, min 1 replica to avoid OCR cold starts), waits for the environment to finish provisioning, and prints the public URL. It is safe to re-run.

**Continuous deployment:** every push to `main` runs [.github/workflows/deploy.yml](.github/workflows/deploy.yml), which builds the image in ACR (tagged with the commit SHA) and updates the container app. It authenticates to Azure with OIDC and needs these repository settings:

- Secrets: `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`
- Variable: `ACR_NAME`

The app runs continuously and incurs Azure charges. To remove everything: `az group delete -n treasury-rg --yes`.

## Files of interest

- `app.py` - FastAPI application
- `ocr_extractor.py` - OCR and extraction logic
- `static/index.html` - web UI
- `tests/` - unit tests (`pytest`)
- `test labels/` - sample label images for manual/end-to-end testing
- `Dockerfile` / `requirements.txt` - container build
- `deploy-azure.ps1` and `.github/workflows/deploy.yml` - Azure deployment

## Notes

- Uses local EasyOCR (models baked into the image) and heuristic regexes; results depend on image quality.
- Fields other than the warning check are simple heuristics and may need refinement.
