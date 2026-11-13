import React from 'react';
import { Users, Ticket, Coffee, ShieldCheck, ArrowUpRight, ArrowDownRight, CheckCircle } from 'lucide-react';

export default function StaffingAdvisor({ forecast }) {
  if (!forecast) return null;

  const upper = forecast.interval_80_upper;
  const occupancy = forecast.predicted_occupancy;
  const capacity = forecast.capacity;

  // Screen Reallocation Recommendation
  let screenRec = {
    title: 'Optimal Screen Allocation',
    description: 'Current auditorium capacity aligns well with projected attendance.',
    icon: CheckCircle,
    color: '#10b981',
    bgColor: '#ecfdf5',
    borderColor: '#a7f3d0'
  };

  if (occupancy >= 0.85 || upper >= capacity) {
    screenRec = {
      title: 'Action Recommended: Upgrade Auditorium Hall',
      description: 'High risk of demand truncation / sell-out. Consider swapping this screening to a larger hall to capture unmet attendance.',
      icon: ArrowUpRight,
      color: '#ef4444',
      bgColor: '#fef2f2',
      borderColor: '#fecaca'
    };
  } else if (occupancy <= 0.20 && capacity > 180) {
    screenRec = {
      title: 'Optimization Opportunity: Downsize to Smaller Screen',
      description: 'Low expected occupancy (<20%). Reallocating to a smaller boutique hall frees the main auditorium for high-traction titles.',
      icon: ArrowDownRight,
      color: '#f59e0b',
      bgColor: '#fffbeb',
      borderColor: '#fde68a'
    };
  }

  // Staffing calculations based on upper bound (protect against 1.5x under-staffing penalty)
  const boxOfficeStaff = Math.max(1, Math.ceil(upper / 150.0));
  const concessionStaff = Math.max(1, Math.ceil(upper / 70.0));
  const usherStaff = Math.max(1, Math.ceil(upper / 120.0));
  const totalStaff = boxOfficeStaff + concessionStaff + usherStaff;

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
        </div>
      </div>

      <h4 style={{ fontSize: '0.96rem', fontWeight: 700, color: '#1e293b', marginBottom: '8px' }}>
        Front-of-House Staffing Allocations
      </h4>
      <p style={{ fontSize: '0.82rem', color: '#64748b', marginBottom: '14px' }}>
        Calibrated against upper 80% bound (<strong>{upper} attendees</strong>) to minimize long concession wait times and usher bottlenecks:
      </p>

      <div className="staffing-cards">
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
      </div>

      <div style={{ textAlign: 'right', marginTop: '12px', fontSize: '0.84rem', fontWeight: 600, color: '#475569' }}>
        Total Recommended Floor Roster: <span style={{ color: '#2563eb' }}>{totalStaff} staff members</span>
      </div>
    </div>
  );
}

