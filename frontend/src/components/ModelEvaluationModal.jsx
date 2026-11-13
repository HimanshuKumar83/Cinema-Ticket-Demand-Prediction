import React from 'react';
import { X, CheckCircle, BarChart3, ShieldAlert } from 'lucide-react';

export default function ModelEvaluationModal({ isOpen, onClose }) {
  if (!isOpen) return null;

  return (
    <div 
      style={{
        position: 'fixed',
        inset: 0,
        backgroundColor: 'rgba(15, 23, 42, 0.6)',
        backdropFilter: 'blur(3px)',
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
          borderRadius: '14px',
          maxWidth: '850px',
          width: '100%',
          maxHeight: '90vh',
          overflowY: 'auto',
          boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1)',
          padding: '28px'
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <div>
            <h2 style={{ fontSize: '1.35rem', fontWeight: 800, color: '#0f172a' }}>
              Held-Out Test Set Model Evaluation
            </h2>
            <p style={{ fontSize: '0.85rem', color: '#64748b' }}>
              Evaluated chronologically on <strong>22,764 shows</strong> (2018-09-30 to 2018-11-04)
            </p>
          </div>
          <button 
            onClick={onClose}
            style={{ border: 'none', background: 'transparent', cursor: 'pointer', padding: '6px' }}
          >
            <X size={22} color="#64748b" />
          </button>
        </div>

        {/* Comparison Table */}
        <div style={{ overflowX: 'auto', marginBottom: '24px' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>MAE (Tickets)</th>
                <th>RMSE</th>
                <th>WAPE (%)</th>
                <th>Occupancy MAE</th>
                <th>Asymmetric Cost</th>
                <th>Clipped (%)</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Hierarchical Baseline</strong></td>
                <td>77.63</td>
                <td>154.01</td>
                <td>62.83%</td>
                <td>0.1148</td>
                <td>89.07</td>
                <td>0.00%</td>
              </tr>
              <tr>
                <td><strong>Ridge Regression (Occupancy)</strong></td>
                <td>67.50</td>
                <td>137.55</td>
                <td>54.63%</td>
                <td>0.1081</td>
                <td>91.84</td>
                <td>0.00%</td>
              </tr>
              <tr style={{ background: '#f0fdf4' }}>
                <td><strong style={{ color: '#166534' }}>LightGBM (Tickets L1) ★</strong></td>
                <td><strong style={{ color: '#166534' }}>48.36</strong></td>
                <td>108.96</td>
                <td><strong>39.14%</strong></td>
                <td>0.0832</td>
                <td><strong>65.66</strong></td>
                <td>0.54%</td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* Slices & Uncertainty */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '18px', marginBottom: '24px' }}>
          <div style={{ background: '#f8fafc', padding: '16px', borderRadius: '10px', border: '1px solid #e2e8f0' }}>
            <h4 style={{ fontSize: '0.92rem', fontWeight: 700, marginBottom: '8px', color: '#1e293b' }}>
              🎯 Cold-Start Performance Slice
            </h4>
            <ul style={{ fontSize: '0.84rem', color: '#475569', listStyleType: 'disc', paddingLeft: '18px' }}>
              <li><strong>Existing Films:</strong> 38.01% WAPE (MAE: 46.2 tickets)</li>
              <li><strong>New Films (Zero Prior Shows):</strong> 73.33% WAPE (MAE: 189.4 tickets)</li>
              <li><strong>Low History (&lt;10 shows):</strong> 73.85% WAPE</li>
            </ul>
          </div>

          <div style={{ background: '#f8fafc', padding: '16px', borderRadius: '10px', border: '1px solid #e2e8f0' }}>
            <h4 style={{ fontSize: '0.92rem', fontWeight: 700, marginBottom: '8px', color: '#1e293b' }}>
              ⚖️ Uncertainty Calibration (80% Interval)
            </h4>
            <ul style={{ fontSize: '0.84rem', color: '#475569', listStyleType: 'disc', paddingLeft: '18px' }}>
              <li><strong>Val Residual Quantiles:</strong> Q10 = -41.2 tickets, Q90 = +36.9 tickets</li>
              <li><strong>Held-out Test Coverage:</strong> 69.28%</li>
              <li>Coverage gap reflects blockbuster opening surges and high capacity variance in late 2018.</li>
            </ul>
          </div>
        </div>

        {/* Limitations Notice */}
        <div style={{ background: '#fffbeb', padding: '14px 18px', borderRadius: '8px', border: '1px solid #fde68a' }}>
          <div style={{ fontWeight: 700, color: '#92400e', fontSize: '0.86rem', display: 'flex', alignItems: 'center', gap: 6 }}>
            <ShieldAlert size={16} /> Documented Operational Limitations:
          </div>
          <p style={{ fontSize: '0.8rem', color: '#78350f', marginTop: 4 }}>
            Dataset spans Feb–Nov 2018 (~8 months, no annual seasonality). Entities and currencies are fully anonymized.
            Recorded data is capped at capacity (censoring latent demand). Predictions are decision-support only.
          </p>
        </div>
      </div>
    </div>
  );
}
