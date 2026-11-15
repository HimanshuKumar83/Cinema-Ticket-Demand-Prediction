"""
CinemaOps AI - Booking Simulator
Simulates realistic, non-linear ticket booking arrival curves, channels, and cancellations
for show-level cinema demand forecasting.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd


@dataclass
class BookingSimulationConfig:
    advance_booking_window_hours: int = 168  # 7 days before showtime
    cancellation_rate: float = 0.03          # 3% cancellation rate
    party_size_dist: Tuple[float, ...] = (0.20, 0.45, 0.20, 0.10, 0.05)  # 1, 2, 3, 4, 5 tickets
    online_channel_share: float = 0.80       # 80% online (web/app), 20% box office walk-ins
    random_seed: int = 42


class BookingSimulator:
    """
    Simulates ticket booking events over time using non-linear arrival curves.
    
    Realistic booking arrival dynamics:
    - Early advance window (T-168h to T-72h): Steady trickle of early planners / movie buffs.
    - Velocity 24h window (T-72h to T-48h): Acceleration begins, especially for weekends/blockbusters.
    - Post T-48h surge (T-48h to T-4h): Significant online bookings.
    - Walk-in / Box office window (T-4h to T-0h): In-person ticket purchases at theater.
    - Hard capacity ceiling: bookings halt once capacity is reached.
    - Small fraction of cancellations occurring before showtime.
    """

    def __init__(self, config: Optional[BookingSimulationConfig] = None):
        self.config = config or BookingSimulationConfig()
        self.rng = np.random.default_rng(self.config.random_seed)

    def simulate_show_bookings(
        self,
        show_id: str,
        show_datetime: pd.Timestamp,
        screen_capacity: int,
        expected_demand_ratio: float,
        is_weekend: bool,
        is_blockbuster: bool,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Simulate timestamped booking events for a single show up to showtime.
        
        Returns:
            Tuple of:
              - list of booking event dicts
              - dict of T-48 point-in-time metrics:
                  'advance_bookings_t48': net tickets booked <= T-48h
                  'booking_velocity_24h': net tickets booked in (T-72h, T-48h]
                  'booking_velocity_previous_24h': net tickets booked in (T-96h, T-72h]
                  'actual_tickets_sold': total net tickets sold by T-0h
        """
        # Determine total latent ticket demand for this show
        noise = self.rng.normal(1.0, 0.12)
        raw_demand = int(round(screen_capacity * expected_demand_ratio * max(0.2, noise)))
        target_demand = min(screen_capacity, max(0, raw_demand))

        if target_demand == 0:
            return [], {
                "advance_bookings_t48": 0,
                "booking_velocity_24h": 0,
                "booking_velocity_previous_24h": 0,
                "actual_tickets_sold": 0,
            }

        # Arrival timing distribution parameters (Beta distribution shifted to [0, 168] hours before show)
        # alpha, beta shape the booking surge.
        # Higher alpha = bookings skew closer to showtime (T=0).
        # Blockbusters and weekends have more advance interest (lower alpha).
        if is_blockbuster:
            alpha = 1.8 if is_weekend else 2.2
            beta_param = 2.0
        else:
            alpha = 2.8 if is_weekend else 3.5
            beta_param = 1.4

        # Generate transaction sizes
        party_sizes = [1, 2, 3, 4, 5]
        events = []
        current_booked_seats = 0
        booking_counter = 0

        # We generate candidate bookings until capacity or demand is reached
        while current_booked_seats < target_demand:
            booking_counter += 1
            party_size = self.rng.choice(party_sizes, p=self.config.party_size_dist)
            seats_to_book = min(party_size, target_demand - current_booked_seats)
            if seats_to_book <= 0:
                break

            # Hours before showtime: Beta sample in [0, 1] mapped to [0, 168] hours before show
            # Value of 0 = showtime, 1 = 168 hours before showtime
            # Note: 1 - Beta sample puts peak closer to showtime
            norm_arrival = self.rng.beta(alpha, beta_param)
            # norm_arrival close to 1 means closer to showtime (hours_before_show close to 0)
            hours_before_show = (1.0 - norm_arrival) * self.config.advance_booking_window_hours
            booking_timestamp = show_datetime - pd.Timedelta(hours=hours_before_show)

            # Channel assignment: Box office only available within 4 hours of show
            if hours_before_show <= 4.0 and self.rng.random() > self.config.online_channel_share:
                channel = "box_office"
            else:
                channel = "mobile_app" if self.rng.random() < 0.65 else "online_web"

            # Cancellations: only for online bookings and > 2 hours before show
            is_cancelled = False
            if channel != "box_office" and hours_before_show > 2.0:
                if self.rng.random() < self.config.cancellation_rate:
                    is_cancelled = True

            event = {
                "booking_id": f"BKG_{show_id}_{booking_counter:04d}",
                "show_id": show_id,
                "booking_timestamp": booking_timestamp,
                "hours_before_show": round(hours_before_show, 2),
                "tickets_booked": seats_to_book,
                "channel": channel,
                "is_cancelled": is_cancelled,
            }
            events.append(event)

            if not is_cancelled:
                current_booked_seats += seats_to_book

        # Sort events chronologically by booking timestamp
        events.sort(key=lambda x: x["booking_timestamp"])

        # Compute point-in-time exact metrics
        cutoff_t48 = show_datetime - pd.Timedelta(hours=48)
        cutoff_t72 = show_datetime - pd.Timedelta(hours=72)
        cutoff_t96 = show_datetime - pd.Timedelta(hours=96)

        advance_bookings_t48 = 0
        booking_velocity_24h = 0
        booking_velocity_previous_24h = 0
        actual_tickets_sold = 0

        for ev in events:
            if ev["is_cancelled"]:
                continue
            t = ev["booking_timestamp"]
            seats = ev["tickets_booked"]

            if t <= cutoff_t48:
                advance_bookings_t48 += seats

            if cutoff_t72 < t <= cutoff_t48:
                booking_velocity_24h += seats

            if cutoff_t96 < t <= cutoff_t72:
                booking_velocity_previous_24h += seats

            if t <= show_datetime:
                actual_tickets_sold += seats

        # Ensure bounds strictly within [0, capacity]
        actual_tickets_sold = min(screen_capacity, actual_tickets_sold)
        advance_bookings_t48 = min(actual_tickets_sold, advance_bookings_t48)

        metrics = {
            "advance_bookings_t48": advance_bookings_t48,
            "booking_velocity_24h": booking_velocity_24h,
            "booking_velocity_previous_24h": booking_velocity_previous_24h,
            "actual_tickets_sold": actual_tickets_sold,
        }

        return events, metrics
