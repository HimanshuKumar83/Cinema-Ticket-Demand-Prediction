# PowerShell script to launch Cinema Ticket Demand Forecasting locally
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   🎬 CinemaOps AI: Demand Forecasting Launch Script     " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Activate Virtual Environment
if (Test-Path .venv\Scripts\Activate.ps1) {
    Write-Host "[1/2] Activating virtual environment (.venv)..." -ForegroundColor Green
    & .\.venv\Scripts\Activate.ps1
} else {
    Write-Host "[!] Virtual environment not found. Please run: python -m venv .venv" -ForegroundColor Red
    exit 1
}

# 2. Check artifacts
if (-not (Test-Path artifacts\model.joblib) -or -not (Test-Path artifacts\history.parquet)) {
    Write-Host "[!] Trained model artifacts missing. Running training pipeline..." -ForegroundColor Yellow
    python -m src.train
}

# 3. Launch Option
Write-Host "`nChoose an interface to run:" -ForegroundColor Yellow
Write-Host "  1) Launch Interactive Streamlit Dashboard (Port 8501) [Recommended]"
Write-Host "  2) Launch FastAPI REST Service (Port 8000)"
Write-Host "  3) Launch Both (API in background, Dashboard in browser)"
Write-Host "  4) Run Pytest Suite"
$choice = Read-Host "`nEnter option [1-4] (Default: 1)"

if ($choice -eq "2") {
    Write-Host "`nStarting FastAPI at http://localhost:8000 (Docs: http://localhost:8000/docs)..." -ForegroundColor Green
    uvicorn app.api:app --reload --port 8000
} elseif ($choice -eq "3") {
    Write-Host "`nStarting FastAPI in background..." -ForegroundColor Green
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; .\.venv\Scripts\Activate.ps1; uvicorn app.api:app --port 8000"
    Start-Sleep -Seconds 2
    Write-Host "Starting Streamlit Dashboard..." -ForegroundColor Green
    streamlit run app/dashboard.py
} elseif ($choice -eq "4") {
    Write-Host "`nRunning Pytest Suite..." -ForegroundColor Green
    pytest -v
} else {
    Write-Host "`nStarting Streamlit Dashboard at http://localhost:8501..." -ForegroundColor Green
    streamlit run app/dashboard.py
}
