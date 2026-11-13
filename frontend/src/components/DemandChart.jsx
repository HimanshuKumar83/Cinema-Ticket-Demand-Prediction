import React from 'react';

export default function DemandChart({ forecast }) {
  if (!forecast) return null;

  const {
    predicted_tickets: tickets,
    interval_80_lower: lower,
    interval_80_upper: upper,
    capacity,
    is_cold_start: isCold
  } = forecast;

  // Compute SVG scale
  const maxAxis = Math.max(capacity * 1.25, upper * 1.25, 100);
  const chartHeight = 220;
  const chartWidth = 520;
  const paddingBottom = 40;
  const paddingTop = 30;
  const usableHeight = chartHeight - paddingBottom - paddingTop;

  const scaleY = (val) => chartHeight - paddingBottom - (val / maxAxis) * usableHeight;

  const barX = 180;
  const barWidth = 80;
  const barY = scaleY(tickets);
  const barHeight = chartHeight - paddingBottom - barY;

  const lowerY = scaleY(lower);
  const upperY = scaleY(upper);
  const capY = scaleY(capacity);

  return (
    <div style={{ background: '#ffffff', borderRadius: '10px', padding: '16px', border: '1px solid #e2e8f0' }}>
      <svg width="100%" height={chartHeight} viewBox={`0 0 ${chartWidth} ${chartHeight}`}>
        {/* Gridlines */}
        {[0, 0.25, 0.5, 0.75, 1.0].map((ratio) => {
          const val = Math.round(maxAxis * ratio);
          const y = scaleY(val);
          return (
            <g key={ratio}>
              <line x1="60" y1={y} x2={chartWidth - 20} y2={y} stroke="#f1f5f9" strokeWidth="1" strokeDasharray="3 3" />
              <text x="50" y={y + 4} textAnchor="end" fontSize="11" fill="#94a3b8">
                {val}
              </text>
            </g>
          );
        })}

        {/* Capacity Limit Line */}
        <line
          x1="60"
          y1={capY}
          x2={chartWidth - 20}
          y2={capY}
          stroke="#ef4444"
          strokeWidth="2"
          strokeDasharray="5 5"
        />
        <text x={chartWidth - 25} y={capY - 6} textAnchor="end" fontSize="11" fill="#ef4444" fontWeight="600">
          Capacity ({capacity} seats)
        </text>

        {/* Forecast Bar */}
        <rect
          x={barX}
          y={barY}
          width={barWidth}
          height={barHeight}
          rx="6"
          fill={isCold ? '#f59e0b' : '#3b82f6'}
          opacity="0.9"
        />

        {/* Error bar / 80% Confidence Interval */}
        <line
          x1={barX + barWidth / 2}
          y1={upperY}
          x2={barX + barWidth / 2}
          y2={lowerY}
          stroke="#0f172a"
          strokeWidth="2.5"
        />
        {/* Upper Whisker */}
        <line
          x1={barX + barWidth / 2 - 16}
          y1={upperY}
          x2={barX + barWidth / 2 + 16}
          y2={upperY}
          stroke="#0f172a"
          strokeWidth="2.5"
        />
        {/* Lower Whisker */}
        <line
          x1={barX + barWidth / 2 - 16}
          y1={lowerY}
          x2={barX + barWidth / 2 + 16}
          y2={lowerY}
          stroke="#0f172a"
          strokeWidth="2.5"
        />

        {/* Value Labels */}
        <text
          x={barX + barWidth / 2}
          y={barY - 8}
          textAnchor="middle"
          fontSize="13"
          fontWeight="700"
          fill="#0f172a"
        >
          {tickets} tickets
        </text>

        <text
          x={barX + barWidth / 2}
          y={chartHeight - 14}
          textAnchor="middle"
          fontSize="12"
          fontWeight="600"
          fill="#64748b"
        >
          Predicted Attendance (80% Interval: [{lower} - {upper}])
        </text>
      </svg>
    </div>
  );
}

