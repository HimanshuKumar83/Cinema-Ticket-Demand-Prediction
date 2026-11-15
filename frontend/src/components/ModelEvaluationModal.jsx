import React, { useState, useEffect } from 'react';
import { X, CheckCircle, BarChart3, ShieldAlert, Award } from 'lucide-react';

export default function ModelEvaluationModal({ isOpen, onClose }) {
  const [benchmark, setBenchmark] = useState([
    { Model: 'Random Forest', MAE: 12.66, RMSE: 17.29, WAPE_pct: 8.68, Occupancy_MAE: 0.0558, Violation_Rate_Pct: 2.73 },
    { Model: 'LightGBM (Tickets L1) ★', MAE: 13.53, RMSE: 18.40, WAPE_pct: 9.28, Occupancy_MAE: 0.0594, Violation_Rate_Pct: 6.47 },
    { Model: 'XGBoost', MAE: 14.21, RMSE: 19.23, WAPE_pct: 9.74, Occupancy_MAE: 0.0621, Violation_Rate_Pct: 4.22 },
    { Model: 'Ridge Regression', MAE: 17.47, RMSE: 22.36, WAPE_pct: 11.98, Occupancy_MAE: 0.0802, Violation_Rate_Pct: 6.33 },
    { Model: 'Hierarchical Baseline', MAE: 45.98, RMSE: 59.68, WAPE_pct: 31.53, Occupancy_MAE: 0.1815, Violation_Rate_Pct: 0.00 },
  ]);

  useEffect(() => {
    async function loadBenchmark() {
      try {
        const res = await fetch('http://localhost:8000/model/benchmark');
        if (res.ok) {
          const data = await res.json();
          if (data && data.benchmark) {
            const formatted = data.benchmark.map((m) => ({
              ...m,
              Model: m.Model === 'LightGBM' ? 'LightGBM (Tickets L1) ★' : m.Model,
            }));
            setBenchmark(formatted);
          }
        }
      } catch (err) {
        // Fallback to initial benchmark
      }
    }
    if (isOpen) {
      loadBenchmark();
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div 
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(15, 23, 42, 0.65)',
        backdropFilter: 'blur(4px)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 50,
        padding: '20px'
      }}
      onClick={onClose}
    >
      <div 
        style={{
          background: 'white',
          borderRadius: '16px',
          maxWidth: '920px',
          width: '100%',
          maxHeight: '90vh',
          overflowY: 'auto',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)',
          padding: '28px'
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '18px' }}>
          <div>
            <h2 style={{ fontSize: '1.4rem', fontWeight: 800, color: '#0f172a', display: 'flex', alignItems: 'center', gap: 8 }}>
              <Award size={24} color="#2563eb" />
              Held-Out Test Set Model Evaluation
            </h2>
            <p style={{ fontSize: '0.86rem', color: '#64748b' }}>
              Rigorous 5-model chronological test comparison (4,500 held-out test shows, strictly zero lookahead leakage).
            </p>
          </div>
          <button 
            onClick={onClose}
            style={{ border: 'none', background: 'transparent', cursor: 'pointer', padding: '6px' }}
          >
            <X size={22} color="#64748b" />
          </button>
        </div>

        {/* 5-Model Comparison Table */}
        <div style={{ overflowX: 'auto', marginBottom: '24px' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>MAE (Tickets)</th>
                <th>RMSE</th>
                <th>WAPE (%)</th>
                <th>Occupancy MAE</th>
                <th>Capacity Violations (%)</th>
              </tr>
            </thead>
            <tbody>
              {benchmark.map((row, idx) => {
                const isSelected = row.Model.includes('LightGBM') || row.Model.includes('Random Forest');
                return (
                  <tr key={idx} style={isSelected ? { background: '#f0fdf4' } : {}}>
                    <td>
                      <strong style={isSelected ? { color: '#166534' } : {}}>{row.Model}</strong>
                    </td>
                    <td><strong>{row.MAE.toFixed(2)}</strong></td>
                    <td>{row.RMSE.toFixed(2)}</td>
                    <td><strong>{row.WAPE_pct.toFixed(2)}%</strong></td>
                    <td>{row.Occupancy_MAE.toFixed(4)}</td>
                    <td>{row.Violation_Rate_Pct ? `${row.Violation_Rate_Pct.toFixed(1)}%` : '0.0%'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Operational Slices & Uncertainty */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '18px', marginBottom: '22px' }}>
          <div style={{ background: '#f8fafc', padding: '16px', borderRadius: '10px', border: '1px solid #e2e8f0' }}>
            <h4 style={{ fontSize: '0.92rem', fontWeight: 700, marginBottom: '8px', color: '#1e293b' }}>
              🎯 Cold-Start Performance Analysis
            </h4>
            <ul style={{ fontSize: '0.84rem', color: '#475569', listStyleType: 'disc', paddingLeft: '18px' }}>
              <li><strong>Established Titles:</strong> High-precision forecasting driven by T-48 velocity.</li>
              <li><strong>Cold-Start Debut Titles:</strong> Fallback to genre and cinema historical demand priors.</li>
              <li><strong>Boutique Halls (&le;150 seats):</strong> Tight capacity bounds prevent overbooking.</li>
            </ul>
          </div>

          <div style={{ background: '#f8fafc', padding: '16px', borderRadius: '10px', border: '1px solid #e2e8f0' }}>
            <h4 style={{ fontSize: '0.92rem', fontWeight: 700, marginBottom: '8px', color: '#1e293b' }}>
              ⚖️ Conformal Prediction Uncertainty (80% Interval)
            </h4>
            <ul style={{ fontSize: '0.84rem', color: '#475569', listStyleType: 'disc', paddingLeft: '18px' }}>
              <li><strong>Validation Residuals:</strong> Q10 = -28.8 tickets, Q90 = +11.3 tickets.</li>
              <li><strong>Empirical Test Coverage:</strong> 79.09% (matches theoretical 80% calibration).</li>
              <li>Interval is strictly clamped within [0, Screen Capacity].</li>
            </ul>
          </div>
        </div>

        {/* Limitations Notice */}
        <div style={{ background: '#fffbeb', padding: '14px 18px', borderRadius: '8px', border: '1px solid #fde68a' }}>
          <div style={{ fontWeight: 700, color: '#92400e', fontSize: '0.86rem', display: 'flex', alignItems: 'center', gap: 6 }}>
            <ShieldAlert size={16} /> Operational Decision-Support Mandate:
          </div>
          <p style={{ fontSize: '0.82rem', color: '#78350f', marginTop: 4 }}>
            System outputs are advisory only and require authorized human managerial review before screen or roster execution. 
            Forecasts generated at T-48h lead time. Unannounced local promotions or weather anomalies must be verified by on-duty cinema managers.
          </p>
        </div>
      </div>
    </div>
  );
}
