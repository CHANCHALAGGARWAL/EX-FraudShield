<#
run_app.ps1

Helper script to activate local .venv (if present), find `app.py`, and start Streamlit.
Usage: Run this from PowerShell in the repo root:
  & .\run_app.ps1
#>

$ErrorActionPreference = 'Stop'

Write-Host "Running run_app.ps1 from: $PSScriptRoot"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Try to activate virtual environment if it exists
$venvActivate = Join-Path $scriptDir ".venv\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    Write-Host "Activating virtual environment: $venvActivate"
    & $venvActivate
} else {
    Write-Warning ".venv activation script not found at $venvActivate. Proceeding with current Python environment."
}

# Locate app.py anywhere under the repo
$appFile = Get-ChildItem -Path $scriptDir -Recurse -Filter app.py -File -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $appFile) {
    Write-Error "Could not find app.py in the repository. Ensure you're running this script from the repo root."
    exit 1
}

Set-Location $appFile.DirectoryName
Write-Host "Starting Streamlit from: $PWD\$($appFile.Name)"

# Start Streamlit (use --server.headless true for automated environments)
streamlit run $appFile.Name --server.port 8501 --server.headless true
