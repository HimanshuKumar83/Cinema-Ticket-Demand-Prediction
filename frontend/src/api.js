/**
 * API client connecting to FastAPI backend
 */
const API_BASE_URL = 'http://localhost:8000';

export async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (!res.ok) throw new Error('Health check failed');
    return await res.json();
  } catch (err) {
    return { status: 'offline', error: err.message };
  }
}

export async function fetchModelInfo() {
  try {
    const res = await fetch(`${API_BASE_URL}/model-info`);
    if (!res.ok) throw new Error('Failed to fetch model info');
    return await res.json();
  } catch (err) {
    return null;
  }
}

export async function requestForecast(payload) {
  try {
    const res = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new Error(errorData.detail || `Server error: ${res.statusText}`);
    }

    return await res.json();
  } catch (err) {
    console.error('Forecast API error:', err);
    throw err;
  }
}

