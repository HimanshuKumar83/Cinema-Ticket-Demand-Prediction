import React from 'react';
import { Calendar, PlayCircle, Clock, Maximize2, Sparkles, TrendingUp, DollarSign } from 'lucide-react';

export default function ForecastForm({
  formData,
  onChange,
  onSubmit,
  loading,
  onApplyPreset,
  activePreset
}) {
  const presets = [
    {
      id: 'blockbuster',
      label: '🔥 Blockbuster Night',
      cinema_id: 'CIN_01',
      movie_id: 'MOV_001',
      date: '2025-01-24',
      show_slot: 'Evening',
      screen_capacity: 350,
      advance_bookings_t48: 130,
      booking_velocity_24h: 55,
      is_weekend: 1,
      is_blockbuster: 1,
      genre: 'Action'
    },
    {
      id: 'matinee',
      label: '☀️ Afternoon Matinee',
      cinema_id: 'CIN_02',
      movie_id: 'MOV_015',
      date: '2025-01-25',
      show_slot: 'Matinee',
      screen_capacity: 200,
      advance_bookings_t48: 45,
      booking_velocity_24h: 18,
      is_weekend: 1,
      is_blockbuster: 0,
      genre: 'Animation'
    },
    {
      id: 'coldstart',
      label: '🆕 Cold-Start Film',
      cinema_id: 'CIN_05',
      movie_id: 'MOV_085',
      date: '2025-02-01',
      show_slot: 'Evening',
      screen_capacity: 220,
      advance_bookings_t48: 25,
      booking_velocity_24h: 10,
      is_weekend: 0,
      is_blockbuster: 0,
      genre: 'Drama'
    },
    {
      id: 'boutique',
      label: '🎪 Boutique 100-Seat',
      cinema_id: 'CIN_10',
      movie_id: 'MOV_008',
      date: '2025-01-26',
      show_slot: 'Night',
      screen_capacity: 100,
      advance_bookings_t48: 55,
      booking_velocity_24h: 22,
      is_weekend: 1,
      is_blockbuster: 0,
      genre: 'Comedy'
    }
  ];

  return (
    <div className="card">
      <h3 className="card-title">
        <PlayCircle size={20} color="#2563eb" />
        Screening Parameters (T-48h Horizon)
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
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
          <div className="form-group">
            <label className="form-label" htmlFor="cinema_code">Cinema Complex Code / ID</label>
            <input
              id="cinema_code"
              type="text"
              className="form-input"
              name="cinema_id"
              value={formData.cinema_id || formData.cinema_code || 'CIN_01'}
              onChange={onChange}
              required
              placeholder="e.g. CIN_01"
            />
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="film_code">Film Code / Movie ID</label>
            <input
              id="film_code"
              type="text"
              className="form-input"
              name="movie_id"
              value={formData.movie_id || formData.film_code || 'MOV_001'}
              onChange={onChange}
              required
              placeholder="e.g. MOV_001"
            />
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
          <div className="form-group">
            <label className="form-label" htmlFor="screening_date">
              <Calendar size={14} style={{ display: 'inline', marginRight: 4, verticalAlign: 'middle' }} />
              Screening Date
            </label>
            <input
              id="screening_date"
              type="date"
              className="form-input"
              name="date"
              value={formData.date || '2025-01-24'}
              onChange={onChange}
              required
            />
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="show_slot">
              <Clock size={14} style={{ display: 'inline', marginRight: 4, verticalAlign: 'middle' }} />
              Show Slot
            </label>
            <select
              id="show_slot"
              className="form-input"
              name="show_slot"
              value={formData.show_slot || 'Evening'}
              onChange={onChange}
            >
              <option value="Morning">Morning (10:00 - 13:00)</option>
              <option value="Matinee">Matinee (13:00 - 17:00)</option>
              <option value="Evening">Evening (17:00 - 20:30)</option>
              <option value="Night">Night (20:30 - 23:30)</option>
            </select>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
          <div className="form-group">
            <label className="form-label" htmlFor="capacity">
              <Maximize2 size={14} style={{ display: 'inline', marginRight: 4, verticalAlign: 'middle' }} />
              Hall Capacity (Seats)
            </label>
            <input
              id="capacity"
              type="number"
              className="form-input"
              name="screen_capacity"
              value={formData.screen_capacity || formData.capacity || 250}
              onChange={onChange}
              required
              min="50"
              max="500"
            />
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="ticket_price">
              <DollarSign size={14} style={{ display: 'inline', marginRight: 4, verticalAlign: 'middle' }} />
              Ticket Price (₹ / $)
            </label>
            <input
              id="ticket_price"
              type="number"
              className="form-input"
              name="ticket_price"
              value={formData.ticket_price || 250}
              onChange={onChange}
              min="50"
              max="1500"
            />
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
          <div className="form-group">
            <label className="form-label" htmlFor="advance_bookings_t48">
              <TrendingUp size={14} style={{ display: 'inline', marginRight: 4, verticalAlign: 'middle' }} />
              Advance Bookings (T-48h)
            </label>
            <input
              id="advance_bookings_t48"
              type="number"
              className="form-input"
              name="advance_bookings_t48"
              value={formData.advance_bookings_t48 !== undefined ? formData.advance_bookings_t48 : 80}
              onChange={onChange}
              min="0"
              max="500"
            />
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="booking_velocity_24h">
              24h Velocity (T-72 to T-48)
            </label>
            <input
              id="booking_velocity_24h"
              type="number"
              className="form-input"
              name="booking_velocity_24h"
              value={formData.booking_velocity_24h !== undefined ? formData.booking_velocity_24h : 35}
              onChange={onChange}
              min="0"
              max="500"
            />
          </div>
        </div>

        <div style={{ display: 'flex', gap: '20px', margin: '10px 0 16px', alignItems: 'center' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.84rem', cursor: 'pointer', color: '#334155' }}>
            <input
              type="checkbox"
              name="is_weekend"
              checked={Boolean(formData.is_weekend)}
              onChange={onChange}
            />
            Weekend Show
          </label>

          <label style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.84rem', cursor: 'pointer', color: '#334155' }}>
            <input
              type="checkbox"
              name="is_blockbuster"
              checked={Boolean(formData.is_blockbuster)}
              onChange={onChange}
            />
            Blockbuster Film
          </label>
        </div>

        <button type="submit" className="btn-primary" disabled={loading}>
          {loading ? 'Forecasting with LightGBM...' : 'Generate T-48h Forecast'}
        </button>
      </form>
    </div>
  );
}
