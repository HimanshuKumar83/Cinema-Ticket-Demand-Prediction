import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import App from '../App.jsx';
import StaffingAdvisor from '../components/StaffingAdvisor.jsx';

describe('CinemaOps AI React Frontend', () => {
  it('renders the header, brand title, and operational governance disclaimer', () => {
    render(<App />);
    expect(screen.getByText('CinemaOps AI')).toBeInTheDocument();
    expect(screen.getByText(/OPERATIONAL DECISION-SUPPORT MANDATE/i)).toBeInTheDocument();
  });

  it('renders fast preset buttons and form inputs', () => {
    render(<App />);
    expect(screen.getByText(/Blockbuster Night/i)).toBeInTheDocument();
    expect(screen.getByText(/Afternoon Matinee/i)).toBeInTheDocument();
    expect(screen.getByText(/Cold-Start Film/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Cinema Complex Code/i)).toBeInTheDocument();
  });

  it('StaffingAdvisor accurately calculates front-of-house staff from upper interval bound', () => {
    const mockForecast = {
      predicted_tickets: 280,
      predicted_occupancy: 0.8,
      capacity: 350,
      interval_80_lower: 240,
      interval_80_upper: 320, // 320 attendees
      is_cold_start: false
    };

    render(<StaffingAdvisor forecast={mockForecast} />);
    // Ushers: ceil(320 / 120) = 3
    // Concessions: ceil(320 / 70) = 5
    // Box Office: ceil(320 / 150) = 3
    expect(screen.getByText('Floor Ushers')).toBeInTheDocument();
    expect(screen.getByText('Concession Staff')).toBeInTheDocument();
    expect(screen.getByText('Box Office / Check-in')).toBeInTheDocument();
  });

  it('opens and closes the Model Benchmarks modal', () => {
    render(<App />);
    const benchmarkBtn = screen.getByText(/Model Benchmarks/i);
    fireEvent.click(benchmarkBtn);
    expect(screen.getByText(/Held-Out Test Set Model Evaluation/i)).toBeInTheDocument();
    expect(screen.getByText(/LightGBM \(Tickets L1\) ★/i)).toBeInTheDocument();
  });
});

