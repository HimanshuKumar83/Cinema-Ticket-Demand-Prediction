import React, { useState, useEffect } from 'react';
import Header from './components/Header.jsx';
import ForecastForm from './components/ForecastForm.jsx';
import DemandChart from './components/DemandChart.jsx';
import StaffingAdvisor from './components/StaffingAdvisor.jsx';
import ModelEvaluationModal from './components/ModelEvaluationModal.jsx';
import { checkBackendHealth, requestForecast } from './api.js';
import { AlertCircle, CheckCircle, Info, Sparkles } from 'lucide-react';

export default function App() {
  const [formData, setFormData] = useState({
    cinema_code: 448,
    film_code: 1576,
    date: '2018-10-15',
    show_time: 4,
    capacity: 350
  });

  const [activePreset, setActivePreset] = useState('blockbuster');
  const [forecast, setForecast] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isOnline, setIsOnline] = useState(false);
  const [isMetricsOpen, setIsMetricsOpen] = useState(false);

  // Check health and run initial prediction on mount
  useEffect(() => {
    async function init() {
      const health = await checkBackendHealth();
      const online = health && health.status === 'healthy';
      setIsOnline(online);

      if (online) {
        handleForecast(formData);
      } else {
        // Fallback default forecast representation if backend not yet started
        setForecast({
          predicted_tickets: 350.0,
          predicted_occupancy: 1.0,
          capacity: 350.0,
          interval_80_lower: 308.8,
          interval_80_upper: 350.0,
          is_cold_start: false,
          cold_start_warning: null,
          top_contributing_features: [
            { feature: 'film_cinema_mean_tickets_14d', value: 3515.93 },
            { feature: 'cinema_slot_mean_tickets_14d', value: 251.89 },
            { feature: 'cinema_mean_tickets_7d_filled', value: 905.85 },
            { feature: 'film_mean_tickets_7d_filled', value: 255.28 },
            { feature: 'capacity', value: 350.0 }
          ],
          limitation_note: 'Decision-support only: human manager review required.'
        });
      }
    }
    init();
  }, []);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: name === 'date' ? value : Number(value)
    }));
    setActivePreset(null);
  };

  const handleApplyPreset = (preset) => {
    const newForm = {
      cinema_code: preset.cinema,
      film_code: preset.film,
      date: preset.date,
      show_time: preset.slot,
      capacity: preset.cap
    };
    setFormData(newForm);
    setActivePreset(preset.id);
    handleForecast(newForm);
  };

  const handleForecast = async (dataToSubmit) => {
    setLoading(true);
    setError(null);
    try {
      const result = await requestForecast(dataToSubmit);
      setForecast(result);
      setIsOnline(true);
    } catch (err) {
      setError(err.message || 'Could not connect to FastAPI backend at http://localhost:8000');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    handleForecast(formData);
  };

  return (
    <div className="app-container">
      <Header isOnline={isOnline} onOpenMetrics={() => setIsMetricsOpen(true)} />

      {error && (
        <div style={{
          background: '#fef2f2',
          border: '1px solid #fecaca',
          borderRadius: '8px',
          padding: '12px 18px',
          color: '#991b1b',
          marginBottom: '20px',
          display: 'flex',
          alignItems: 'center',
          gap: '10px'
        }}>
          <AlertCircle size={20} />
          <div>
            <strong>Backend Connection Notice:</strong> {error}. Ensure FastAPI is running (`uvicorn app.api:app --reload --port 8000`).
          </div>
        </div>
      )}

      <div className="main-grid">
        {/* Left Column: Form & Presets */}
        <div>
          <ForecastForm
            formData={formData}
            onChange={handleInputChange}
            onSubmit={handleSubmit}
            loading={loading}
            onApplyPreset={handleApplyPreset}
            activePreset={activePreset}
          />
        </div>

        {/* Right Column: Dynamic KPIs, Chart, Advisor & Features */}
        <div>
          {forecast && (
            <>
              {/* Cold-Start Diagnostic Banner */}
              {forecast.is_cold_start ? (
                <div className="alert-card warning">
                  <AlertCircle size={24} style={{ flexShrink: 0 }} />
                  <div>
                    <div style={{ fontWeight: 700, fontSize: '0.94rem' }}>Cold-Start / Low-History Screening</div>
                    <div style={{ fontSize: '0.84rem', marginTop: 2 }}>
                      {forecast.cold_start_warning || 'Film has minimal prior screening data. Relying on slot and cinema historical priors.'}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="alert-card success">
                  <CheckCircle size={24} style={{ flexShrink: 0 }} />
                  <div>
                    <div style={{ fontWeight: 700, fontSize: '0.94rem' }}>Established Screening Data Available</div>
                    <div style={{ fontSize: '0.84rem', marginTop: 2 }}>
                      Robust historical tracking found for Film {formData.film_code} at Cinema {formData.cinema_code}.
                    </div>
                  </div>
                </div>
              )}

              {/* Metric Cards Ribbon */}
              <div className="metrics-grid">
                <div className="metric-tile">
                  <div className="metric-label">Predicted Tickets</div>
                  <div className="metric-value">{Math.round(forecast.predicted_tickets)}</div>
                  <div className="metric-sub">Expected attendance</div>
                </div>

                <div className="metric-tile">
                  <div className="metric-label">Projected Occupancy</div>
                  <div className="metric-value">{(forecast.predicted_occupancy * 100).toFixed(1)}%</div>
                  <div className="metric-sub">{Math.round(forecast.predicted_tickets)} / {forecast.capacity} seats</div>
                </div>

                <div className="metric-tile">
                  <div className="metric-label">80% Prediction Window</div>
                  <div className="metric-value" style={{ fontSize: '1.45rem' }}>
                    [{Math.round(forecast.interval_80_lower)} - {Math.round(forecast.interval_80_upper)}]
                  </div>
                  <div className="metric-sub">Residual quantile bounds</div>
                </div>

                <div className="metric-tile">
                  <div className="metric-label">Hall Capacity</div>
                  <div className="metric-value">{forecast.capacity}</div>
                  <div className="metric-sub">Room ceiling</div>
                </div>
              </div>

              {/* Demand Chart Card */}
              <div className="card">
                <h3 className="card-title">
                  <Sparkles size={20} color="#2563eb" />
                  Attendance Demand vs Room Capacity
                </h3>
                <DemandChart forecast={forecast} />
              </div>

              {/* Operational Staffing & Hall Reallocation Advisor */}
              <div className="card">
                <StaffingAdvisor forecast={forecast} />
              </div>

              {/* Local Input Drivers Table */}
              {forecast.top_contributing_features && forecast.top_contributing_features.length > 0 && (
                <div className="card">
                  <h3 className="card-title">
                    <Info size={20} color="#64748b" />
                    Key Historical Features for this Slot
                  </h3>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Input Feature Name</th>
                        <th>Historical Observed Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {forecast.top_contributing_features.map((f, i) => (
                        <tr key={i}>
                          <td><code>{f.feature}</code></td>
                          <td><strong>{typeof f.value === 'number' ? f.value.toLocaleString() : String(f.value)}</strong></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <div style={{ fontSize: '0.78rem', color: '#64748b', marginTop: 10 }}>
                    *All features computed strictly from dates &le; show_date - 2 days (T-48h horizon).
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      <ModelEvaluationModal isOpen={isMetricsOpen} onClose={() => setIsMetricsOpen(false)} />
    </div>
  );
}
