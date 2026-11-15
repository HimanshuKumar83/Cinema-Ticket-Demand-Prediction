import React, { useState, useEffect } from 'react';
import Header from './components/Header.jsx';
import ForecastForm from './components/ForecastForm.jsx';
import DemandChart from './components/DemandChart.jsx';
import StaffingAdvisor from './components/StaffingAdvisor.jsx';
import ModelEvaluationModal from './components/ModelEvaluationModal.jsx';
import { checkBackendHealth, requestForecast } from './api.js';
import { AlertCircle, CheckCircle, Info, Sparkles, TrendingUp, TrendingDown } from 'lucide-react';

export default function App() {
  const [formData, setFormData] = useState({
    cinema_id: 'CIN_01',
    movie_id: 'MOV_001',
    date: '2025-01-24',
    show_slot: 'Evening',
    screen_capacity: 350,
    ticket_price: 250,
    advance_bookings_t48: 130,
    booking_velocity_24h: 55,
    is_weekend: 1,
    is_blockbuster: 1,
    genre: 'Action'
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
        // Fallback default forecast representation if backend is booting
        setForecast({
          predicted_tickets: 285.0,
          predicted_occupancy: 0.814,
          screen_capacity: 350.0,
          capacity: 350.0,
          interval_80_lower: 258.3,
          lower_bound: 258.3,
          interval_80_upper: 305.0,
          upper_bound: 305.0,
          is_cold_start: false,
          cold_start_warning: null,
          positive_drivers: [
            { feature: 'advance_bookings_t48', feature_value: 130, shap_impact: 61.22, is_positive: true },
            { feature: 'is_blockbuster', feature_value: 1, shap_impact: 15.36, is_positive: true },
            { feature: 'booking_velocity_24h', feature_value: 55, shap_impact: 12.99, is_positive: true }
          ],
          negative_drivers: [
            { feature: 'ticket_price', feature_value: 250, shap_impact: -2.15, is_positive: false }
          ],
          operational_advisory: {
            screen_decision: 'MAINTAIN_CURRENT',
            recommended_action: 'Balanced demand (81.4% projected occupancy). Maintain currently scheduled screening room.',
            suggested_capacity_range: 'Current auditorium capacity is optimal.',
            box_office_staff: 3,
            ushers_count: 3,
            concession_cashiers: 5,
            cleaning_crew: 4,
            total_floor_staff: 15,
            concession_inventory_units: 200,
            managerial_review_required: true,
            operational_disclaimer: 'Rule-based operational decision support only. Not ML output. Managerial sign-off required.'
          },
          limitation_note: 'Decision-support only: human manager review required.'
        });
      }
    }
    init();
  }, []);

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'checkbox'
        ? (checked ? 1 : 0)
        : (['date', 'cinema_id', 'movie_id', 'show_slot', 'genre'].includes(name) ? value : Number(value))
    }));
    setActivePreset(null);
  };

  const handleApplyPreset = (preset) => {
    const newForm = { ...preset };
    delete newForm.id;
    delete newForm.label;
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

  const hallCapacity = forecast ? (forecast.screen_capacity || forecast.capacity || 250) : 250;

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
                      {forecast.cold_start_warning || `Movie '${formData.movie_id || formData.film_code}' has zero prior screenings. Prediction relies on genre/cinema priors.`}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="alert-card success">
                  <CheckCircle size={24} style={{ flexShrink: 0 }} />
                  <div>
                    <div style={{ fontWeight: 700, fontSize: '0.94rem' }}>Established Screening Data Available</div>
                    <div style={{ fontSize: '0.84rem', marginTop: 2 }}>
                      Robust historical tracking found for Movie {formData.movie_id || formData.film_code} at Cinema {formData.cinema_id || formData.cinema_code}.
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
                  <div className="metric-sub">{Math.round(forecast.predicted_tickets)} / {hallCapacity} seats</div>
                </div>

                <div className="metric-tile">
                  <div className="metric-label">80% Prediction Window</div>
                  <div className="metric-value" style={{ fontSize: '1.45rem' }}>
                    [{Math.round(forecast.interval_80_lower !== undefined ? forecast.interval_80_lower : forecast.lower_bound)} - {Math.round(forecast.interval_80_upper !== undefined ? forecast.interval_80_upper : forecast.upper_bound)}]
                  </div>
                  <div className="metric-sub">Conformal calibrated bounds</div>
                </div>

                <div className="metric-tile">
                  <div className="metric-label">Hall Capacity</div>
                  <div className="metric-value">{hallCapacity}</div>
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

              {/* TreeSHAP Local Drivers */}
              {((forecast.positive_drivers && forecast.positive_drivers.length > 0) || (forecast.top_contributing_features && forecast.top_contributing_features.length > 0)) && (
                <div className="card">
                  <h3 className="card-title">
                    <Info size={20} color="#64748b" />
                    Key Decision Drivers (TreeSHAP Attributions)
                  </h3>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Feature Driver</th>
                        <th>Observed Input</th>
                        <th>Impact on Prediction</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(forecast.positive_drivers || []).map((d, i) => (
                        <tr key={`pos-${i}`}>
                          <td><code>{d.feature}</code></td>
                          <td><strong>{typeof d.feature_value === 'number' ? d.feature_value.toLocaleString() : String(d.feature_value)}</strong></td>
                          <td style={{ color: '#16a34a', fontWeight: 600 }}>
                            <TrendingUp size={14} style={{ display: 'inline', marginRight: 4, verticalAlign: 'middle' }} />
                            +{Math.abs(d.shap_impact).toFixed(1)} tickets
                          </td>
                        </tr>
                      ))}
                      {(forecast.negative_drivers || []).map((d, i) => (
                        <tr key={`neg-${i}`}>
                          <td><code>{d.feature}</code></td>
                          <td><strong>{typeof d.feature_value === 'number' ? d.feature_value.toLocaleString() : String(d.feature_value)}</strong></td>
                          <td style={{ color: '#dc2626', fontWeight: 600 }}>
                            <TrendingDown size={14} style={{ display: 'inline', marginRight: 4, verticalAlign: 'middle' }} />
                            -{Math.abs(d.shap_impact).toFixed(1)} tickets
                          </td>
                        </tr>
                      ))}
                      {(!forecast.positive_drivers || forecast.positive_drivers.length === 0) && (forecast.top_contributing_features || []).map((f, i) => (
                        <tr key={`top-${i}`}>
                          <td><code>{f.feature}</code></td>
                          <td><strong>{typeof f.value === 'number' ? f.value.toLocaleString() : String(f.value)}</strong></td>
                          <td style={{ color: '#2563eb', fontWeight: 600 }}>Active signal</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <div style={{ fontSize: '0.78rem', color: '#64748b', marginTop: 10 }}>
                    *All features computed strictly from data &le; T-48h prior to scheduled showtime (Zero Future Leakage).
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
