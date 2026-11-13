import React from 'react';
import { Film, Activity, AlertTriangle } from 'lucide-react';

export default function Header({ isOnline, onOpenMetrics }) {
  return (
    <>
      <header className="header-bar">
        <div className="brand-section">
          <div className="brand-icon">
            <Film size={24} />
          </div>
          <div>
            <h1 className="brand-title">CinemaOps AI</h1>
            <p className="brand-subtitle">Show-Level Ticket Demand Forecasting & Staffing Optimization (T-48h)</p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <button 
            onClick={onOpenMetrics}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '8px 16px',
              borderRadius: '8px',
              border: '1px solid #cbd5e1',
              background: '#ffffff',
              fontSize: '0.84rem',
              fontWeight: 600,
              cursor: 'pointer',
              color: '#334155'
            }}
          >
            <Activity size={16} color="#2563eb" />
            Model Benchmarks
          </button>

          <div className={`status-badge ${isOnline ? 'online' : 'offline'}`}>
            <span className="pulse-dot" style={{ backgroundColor: isOnline ? '#10b981' : '#ef4444' }} />
            {isOnline ? 'FastAPI Connected (Port 8000)' : 'API Disconnected'}
          </div>
        </div>
      </header>

      <div className="disclaimer-banner">
        <AlertTriangle size={20} style={{ flexShrink: 0, marginTop: 2 }} />
        <div>
          <strong>OPERATIONAL DECISION-SUPPORT MANDATE:</strong> Human managerial review required.
          Forecasts are advisory and computed at a strict T-48h lookahead horizon.
          Managers must verify screen and staff allocations before committing schedules.
        </div>
      </div>
    </>
  );
}
