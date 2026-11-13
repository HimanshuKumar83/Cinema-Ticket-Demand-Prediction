# Deployment Guide: Cinema Ticket Demand Forecasting MVP

This guide outlines deployment options ranging from local execution on Windows to 100% free cloud deployment platforms.

---

## 💻 1. Local Execution (Windows PowerShell)

You can launch the frontend and backend with a single command:

```powershell
.\run_local.ps1
```

Or run the components directly:

### Run the Interactive Dashboard:
```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app/dashboard.py
```
- Opens automatically in your browser at: `http://localhost:8501`

### Run the FastAPI Backend:
```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.api:app --reload --port 8000
```
- API Base: `http://localhost:8000`
- Interactive Swagger UI: `http://localhost:8000/docs`
- Health Probe: `http://localhost:8000/health`

---

## ☁️ 2. Free Cloud Deployment: Streamlit Community Cloud (Recommended)

Streamlit Community Cloud provides **100% free, zero-config hosting** directly from your GitHub repository.

### Step-by-Step Instructions:
1. **Push your code to GitHub:**
   ```powershell
   git remote add origin https://github.com/<your-username>/Cinema_Ticket.git
   git branch -M main
   git push -u origin main
   ```
   *(Note: Ensure `artifacts/model.joblib` and `artifacts/history.parquet` are pushed so the app has its trained model and historical lookup table).*

2. **Deploy on Streamlit Community Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
   - Click **"New app"**.
   - Select your repository: `<your-username>/Cinema_Ticket`.
   - Set **Main file path**: `app/dashboard.py`.
   - Click **"Deploy!"**.

3. **Your Live Dashboard:**
   - In ~2 minutes, your dashboard will be live at a public URL: `https://<your-app-name>.streamlit.app`
   - Shareable with assessors, hiring managers, and teammates!

---

## 🤗 3. Free Cloud Deployment: Hugging Face Spaces

Hugging Face Spaces offers free CPU hardware and permanent hosting for machine learning demos.

### Step-by-Step Instructions:
1. Log in to [Hugging Face](https://huggingface.co/) and navigate to **Spaces**.
2. Click **"Create new Space"**:
   - **Space Name:** `cinema-ticket-demand-forecaster`
   - **License:** `CC-BY-NC-SA-4.0` (matching source dataset)
   - **Space SDK:** Choose **Streamlit** (or **Docker**).
3. Connect your GitHub repository or push directly via git:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/cinema-ticket-demand-forecaster
   git push space main
   ```
4. If using Streamlit SDK, ensure `app/dashboard.py` is the entry point (or create a root `app.py` that imports `app/dashboard.py`).
5. Your Space builds automatically and runs live.

---

## 🐳 4. Containerized Deployment (Docker & Docker Compose)

The repository includes a production-ready `Dockerfile` and `docker-compose.yml`.

### Build & Run Both API and Dashboard:
```bash
# Build and run containers in background
docker compose up -d --build
```
- **Streamlit Dashboard:** `http://localhost:8501`
- **FastAPI REST API:** `http://localhost:8000`
- **Swagger Docs:** `http://localhost:8000/docs`

### Stop Containers:
```bash
docker compose down
```

---

## 🚀 5. Production Cloud Hosting (Render / Railway)

Both [Render](https://render.com) and [Railway](https://railway.app) offer simple container hosting:

1. Connect your GitHub repository to Render/Railway.
2. Select **"Docker"** as the build runtime.
3. Set the environment port:
   - For Dashboard: `PORT=8501`
   - For API: `PORT=8000`
4. The service will build from `Dockerfile` and expose a secure HTTPS URL automatically.
