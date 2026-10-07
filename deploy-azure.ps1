# Deploy to Azure Container Apps (builds remotely in ACR; no local Docker needed).
# Prereqs: az login; az extension add --name containerapp
# Safe to re-run: reuses the existing registry in the resource group if one exists.
param(
    [string]$ResourceGroup = "treasury-rg",
    [string]$Location = "eastus",
    [string]$AppName = "label-ocr",
    [string]$Registry = ""
)
$ErrorActionPreference = "Stop"

function Invoke-Az {
    # Native commands don't honor $ErrorActionPreference, so check the exit code explicitly.
    & az @args
    if ($LASTEXITCODE -ne 0) { throw "az $($args -join ' ') failed with exit code $LASTEXITCODE" }
}

Invoke-Az group create -n $ResourceGroup -l $Location -o none

if (-not $Registry) {
    $Registry = az acr list -g $ResourceGroup --query "[0].name" -o tsv
    if (-not $Registry) { $Registry = "treasuryacr$((Get-Random -Maximum 99999))" }
}
if (-not (az acr show -n $Registry -g $ResourceGroup --query name -o tsv 2>$null)) {
    Invoke-Az acr create -n $Registry -g $ResourceGroup --sku Basic --admin-enabled true -o none
}
Invoke-Az acr build -r $Registry -t "${AppName}:latest" .

$envName = "$AppName-env"
if (-not (az containerapp env show -n $envName -g $ResourceGroup --query name -o tsv 2>$null)) {
    Invoke-Az containerapp env create -n $envName -g $ResourceGroup -l $Location -o none
}

# Environment provisioning can take several minutes; creating the app before it finishes fails.
$deadline = (Get-Date).AddMinutes(20)
while ($true) {
    $state = az containerapp env show -n $envName -g $ResourceGroup --query properties.provisioningState -o tsv
    Write-Host "Environment state: $state"
    if ($state -eq "Succeeded") { break }
    if ($state -in @("Failed", "Canceled")) { throw "Environment provisioning ended in state '$state'." }
    if ((Get-Date) -gt $deadline) { throw "Timed out waiting for environment (last state '$state')." }
    Start-Sleep -Seconds 30
}

Invoke-Az containerapp up -n $AppName -g $ResourceGroup --environment $envName `
    --image "$Registry.azurecr.io/${AppName}:latest" --registry-server "$Registry.azurecr.io" `
    --target-port 8000 --ingress external
Invoke-Az containerapp update -n $AppName -g $ResourceGroup --cpu 2 --memory 4Gi --min-replicas 1 -o none
az containerapp show -n $AppName -g $ResourceGroup --query properties.configuration.ingress.fqdn -o tsv
