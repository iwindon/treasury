# Deploy to Azure Container Apps (builds remotely in ACR; no local Docker needed).
# Prereqs: az login; az extension add --name containerapp
param(
    [string]$ResourceGroup = "treasury-rg",
    [string]$Location = "eastus",
    [string]$AppName = "label-ocr",
    [string]$Registry = "treasuryacr$((Get-Random -Maximum 99999))"
)
$ErrorActionPreference = "Stop"
az group create -n $ResourceGroup -l $Location | Out-Null
az acr create -n $Registry -g $ResourceGroup --sku Basic --admin-enabled true | Out-Null
az acr build -r $Registry -t "${AppName}:latest" .
az containerapp env create -n "$AppName-env" -g $ResourceGroup -l $Location | Out-Null
az containerapp up -n $AppName -g $ResourceGroup --environment "$AppName-env" `
    --image "$Registry.azurecr.io/${AppName}:latest" --registry-server "$Registry.azurecr.io" `
    --target-port 8000 --ingress external
az containerapp update -n $AppName -g $ResourceGroup --cpu 2 --memory 4Gi --min-replicas 1
az containerapp show -n $AppName -g $ResourceGroup --query properties.configuration.ingress.fqdn -o tsv
