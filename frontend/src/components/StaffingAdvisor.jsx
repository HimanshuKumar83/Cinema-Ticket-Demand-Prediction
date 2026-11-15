import React from 'react';
import { Users, Ticket, Coffee, Sparkles, ArrowUpRight, ArrowDownRight, CheckCircle, ShieldAlert, Brush } from 'lucide-react';

export default function StaffingAdvisor({ forecast }) {
  if (!forecast) return null;

  const advisory = forecast.operational_advisory || {};
  const capacity = forecast.screen_capacity || forecast.capacity || 250;
  const upper = forecast.interval_80_upper !== undefined ? forecast.interval_80_upper : (forecast.upper_bound !== undefined ? forecast.upper_bound : capacity);
  const occupancy = forecast.predicted_occupancy || 0;

  // Screen Reallocation Recommendation from centralized advisory or fallback
  const decision = advisory.screen_decision || (
    occupancy >= 0.85 || upper >= capacity
      ? 'UPGRADE_SCREEN'
      : (occupancy <= 0.35 && capacity > 180 ? 'DOWNGRADE_SCREEN' : 'MAINTAIN_CURRENT')
  );

  let screenRec = {
    title: 'Optimal Screen Allocation',
    description: advisory.recommended_action || 'Current auditorium capacity aligns well with projected attendance.',
    icon: CheckCircle,
    color: '#10b981',
    bgColor: '#ecfdf5',
    borderColor: '#a7f3d0'
  };

  if (decision === 'UPGRADE_SCREEN') {
    screenRec = {
      title: 'Action Recommended: Upgrade Auditorium Hall',
      description: advisory.recommended_action || 'High risk of demand truncation / sell-out. Reallocate to a larger auditorium screen (350+ seats) to capture unmet demand.',
      icon: ArrowUpRight,
      color: '#ef4444',
      bgColor: '#fef2f2',
      borderColor: '#fecaca'
    };
  } else if (decision === 'DOWNGRADE_SCREEN') {
    screenRec = {
      title: 'Optimization Opportunity: Downsize to Smaller Screen',
      description: advisory.recommended_action || 'Auditorium under-utilized. Shift to a compact theater room (80-150 seats) to minimize HVAC & operational costs.',
      icon: ArrowDownRight,
      color: '#f59e0b',
      bgColor: '#fffbeb',
      borderColor: '#fde68a'
    };
  }

  // Authoritative staffing allocations from backend advisory
  const boxOfficeStaff = advisory.box_office_staff || Math.max(1, Math.ceil(upper / 150.0));
  const concessionStaff = advisory.concession_cashiers || Math.max(1, Math.ceil(upper / 70.0));
  const usherStaff = advisory.ushers_count || Math.max(1, Math.ceil(upper / 120.0));
  const cleaningStaff = advisory.cleaning_crew || Math.max(1, Math.ceil(capacity / 100.0));
  const totalStaff = advisory.total_floor_staff || (boxOfficeStaff + concessionStaff + usherStaff + cleaningStaff);
  const concessionUnits = advisory.concession_inventory_units || Math.round((forecast.predicted_tickets || 0) * 0.70);
  const disclaimer = advisory.operational_disclaimer || 'Rule-based operational decision support only. Not ML output. Managerial sign-off required.';

  const IconComponent = screenRec.icon;

  return (
    <div className="rec-box">
      {/* Screen Allocation Alert */}
      <div 
        style={{
          display: 'flex',
          alignItems: 'flex-start',
          gap: '12px',
          padding: '14px 18px',
          borderRadius: '8px',
          background: screenRec.bgColor,
          border: `1px solid ${screenRec.borderColor}`,
          color: screenRec.color,
          marginBottom: '16px'
        }}
      >
        <IconComponent size={22} style={{ flexShrink: 0, marginTop: 2 }} />
        <div>
          <div style={{ fontWeight: 700, fontSize: '0.94rem' }}>{screenRec.title}</div>
          <div style={{ fontSize: '0.84rem', marginTop: 2, color: '#334155' }}>
            {screenRec.description}
          </div>
          {advisory.suggested_capacity_range && (
            <div style={{ fontSize: '0.78rem', marginTop: 4, fontWeight: 600, color: '#475569' }}>
              Suggested Capacity: {advisory.suggested_capacity_range}
            </div>
          )}
        </div>
      </div>

      <h4 style={{ fontSize: '0.96rem', fontWeight: 700, color: '#1e293b', marginBottom: '8px' }}>
        Front-of-House Staffing Allocations (Risk-Calibrated)
      </h4>
      <p style={{ fontSize: '0.82rem', color: '#64748b', marginBottom: '14px' }}>
        Staffing planned against upper 80% prediction bound (<strong>{Math.round(upper)} attendees</strong>) to minimize asymmetric under-staffing penalty (1.5x) and queue delays:
      </p>

      <div className="staffing-cards" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '10px' }}>
        <div className="staff-card">
          <Ticket size={20} color="#2563eb" style={{ margin: '0 auto 6px' }} />
          <div className="staff-count">{boxOfficeStaff}</div>
          <div className="staff-role">Box Office / Check-in</div>
          <div style={{ fontSize: '0.74rem', color: '#94a3b8', marginTop: 4 }}>1 per 150 guests</div>
        </div>

        <div className="staff-card">
          <Coffee size={20} color="#f59e0b" style={{ margin: '0 auto 6px' }} />
          <div className="staff-count">{concessionStaff}</div>
          <div className="staff-role">Concession Staff</div>
          <div style={{ fontSize: '0.74rem', color: '#94a3b8', marginTop: 4 }}>1 per 70 guests</div>
        </div>

        <div className="staff-card">
          <Users size={20} color="#10b981" style={{ margin: '0 auto 6px' }} />
          <div className="staff-count">{usherStaff}</div>
          <div className="staff-role">Floor Ushers</div>
          <div style={{ fontSize: '0.74rem', color: '#94a3b8', marginTop: 4 }}>1 per 120 guests</div>
        </div>

        <div className="staff-card">
          <Brush size={20} color="#8b5cf6" style={{ margin: '0 auto 6px' }} />
          <div className="staff-count">{cleaningStaff}</div>
          <div className="staff-role">Cleaning Crew</div>
          <div style={{ fontSize: '0.74rem', color: '#94a3b8', marginTop: 4 }}>1 per 100 seats</div>
        </div>
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '16px', flexWrap: 'wrap', gap: '10px' }}>
        <div style={{ fontSize: '0.84rem', color: '#475569' }}>
          🍿 <strong>Concession Inventory:</strong> Recommend preparing ~<strong>{concessionUnits}</strong> combos (70% attachment).
        </div>
        <div style={{ fontSize: '0.84rem', fontWeight: 700, color: '#1e293b' }}>
          Total Floor Staff: <span style={{ color: '#2563eb' }}>{totalStaff} employees</span>
        </div>
      </div>

      {/* Operational Governance Mandate Banner */}
      <div style={{
        marginTop: '14px',
        padding: '10px 14px',
        background: '#f8fafc',
        border: '1px solid #e2e8f0',
        borderRadius: '6px',
        fontSize: '0.76rem',
        color: '#64748b',
        display: 'flex',
        alignItems: 'center',
        gap: '8px'
      }}>
        <ShieldAlert size={16} color="#64748b" style={{ flexShrink: 0 }} />
        <span>{disclaimer}</span>
      </div>
    </div>
  );
}
