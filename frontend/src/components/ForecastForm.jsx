import React from 'react';
import { Calendar, PlayCircle, Clock, Maximize2, Sparkles } from 'lucide-react';

export default function ForecastForm({
  formData,
  onChange,
  onSubmit,
  loading,
  onApplyPreset,
  activePreset
}) {
  const presets = [
    { id: 'blockbuster', label: '🔥 Blockbuster Night', cinema: 448, film: 1576, date: '2018-10-15', slot: 4, cap: 350 },
    { id: 'matinee', label: '☀️ Afternoon Matinee', cinema: 448, film: 1576, date: '2018-10-16', slot: 1, cap: 200 },
    { id: 'coldstart', label: '🆕 Cold-Start Film', cinema: 448, film: 99999, date: '2018-10-15', slot: 3, cap: 250 },
    { id: 'boutique', label: '🎪 Boutique 100-Seat', cinema: 100, film: 1576, date: '2018-10-15', slot: 3, cap: 100 }
  ];

  return (
    <div className="card">
      <h3 className="card-title">
        <PlayCircle size={20} color="#2563eb" />
        Screening Parameters (T-48h)
      </h3>

      {/* Preset Buttons */}
      <div style={{ marginBottom: '14px' }}>
        <span style={{ fontSize: '0.78rem', fontWeight: 600, color: '#64748b', display: 'flex', alignItems: 'center', gap: 4, marginBottom: 8 }}>
          <Sparkles size={14} color="#f59e0b" /> Fast Test Presets:
        </span>
        <div className="presets-container">
          {presets.map((p) => (
            <button
              key={p.id}
              type="button"
              className={`preset-pill ${activePreset === p.id ? 'active' : ''}`}
              onClick={() => onApplyPreset(p)}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      <form onSubmit={onSubmit}>
        <div className="form-group">
          <label className="form-label" htmlFor="cinema_code">Cinema Complex Code</label>
          <input
            id="cinema_code"
            type="number"
            className="form-input"
            name="cinema_code"
            value={formData.cinema_code}
            onChange={onChange}
            required
            min="1"
          />
        </div>

        <div className="form-group">
          <label className="form-label" htmlFor="film_code">Film Code</label>
          <input
            id="film_code"
            type="number"
            className="form-input"
            name="film_code"
            value={formData.film_code}
            onChange={onChange}
            required
            min="1"
          />
        </div>

        <div className="form-group">
          <label className="form-label" htmlFor="screening_date">
            <Calendar size={14} style={{ display: 'inline', marginRight: 4, verticalAlign: 'middle' }} />
            Screening Date (T-48h Horizon)
          </label>
          <input
            id="screening_date"
            type="date"
            className="form-input"
            name="date"
            value={formData.date}
            onChange={onChange}
            required
          />
          <span style={{ fontSize: '0.74rem', color: '#64748b', display: 'block', marginTop: 4 }}>
            Forecast uses past data strictly &le; Date - 2 days.
          </span>
        </div>

        <div className="form-group">
          <label className="form-label" htmlFor="show_time">
            <Clock size={14} style={{ display: 'inline', marginRight: 4, verticalAlign: 'middle' }} />
            Show Time Slot (1 = Matinee, 2 = Afternoon, 3 = Evening)
          </label>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <input
              id="show_time"
              type="range"
              min="1"
              max="12"
              name="show_time"
              value={formData.show_time}
              onChange={onChange}
              style={{ flex: 1 }}
            />
            <span style={{ fontWeight: 700, minWidth: '40px', textAlign: 'center' }}>
              Slot {formData.show_time}
            </span>
          </div>
        </div>

        <div className="form-group">
          <label className="form-label" htmlFor="capacity">
            <Maximize2 size={14} style={{ display: 'inline', marginRight: 4, verticalAlign: 'middle' }} />
            Screen Room Capacity (Seats)
          </label>
          <input
            id="capacity"
            type="number"
            className="form-input"
            name="capacity"
            value={formData.capacity}
            onChange={onChange}
            required
            min="10"
            max="5000"
          />
        </div>

        <button type="submit" className="btn-primary" disabled={loading}>
          {loading ? 'Forecasting with LightGBM...' : 'Generate T-48h Forecast'}
        </button>
      </form>
    </div>
  );
}
