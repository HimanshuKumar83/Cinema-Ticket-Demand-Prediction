"""
CinemaOps AI - Operational Advisory Engine
Centralized, rule-based operational decision-support logic for screen allocation
and front-of-house staffing rosters based on capacity and prediction intervals.

NOTE: All outputs are rule-based operational advice, NOT raw ML model outputs.
Managerial sign-off is mandatory before executing screen changes or staffing shifts.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class StaffingBreakdown:
    box_office_staff: int
    concession_cashiers: int
    floor_ushers: int
    cleaning_crew: int
    total_floor_staff: int
    concession_inventory_units: int
    planning_basis: str
    role_ratios: Dict[str, str]


@dataclass
class ScreenAllocationDecision:
    decision: str
    action: str
    suggested_capacity_range: str
    occupancy_threshold_basis: str


@dataclass
class OperationalAdvisory:
    screen_decision: str
    recommended_action: str
    suggested_capacity_range: str
    box_office_staff: int
    ushers_count: int
    concession_cashiers: int
    cleaning_crew: int
    total_floor_staff: int
    concession_inventory_units: int
    staffing_details: Dict[str, Any]
    screen_details: Dict[str, Any]
    managerial_review_required: bool
    operational_disclaimer: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_staffing_and_screen_advisory(
    predicted_tickets: float,
    predicted_occupancy: float,
    screen_capacity: float,
    interval_80_upper: float,
) -> OperationalAdvisory:
    """
    Calculate centralized operational recommendations:
    1. Screen Capacity Reallocation:
       - Upgrade: Occupancy >= 85% or Upper bound >= Capacity (truncation risk)
       - Downgrade: Occupancy <= 35% and Capacity > 180 (under-utilization)
       - Maintain: 35% < Occupancy < 85%
    
    2. Front-of-House Staffing:
       - Planned against upper 80% prediction bound to protect against 1.5x asymmetric under-staffing penalty.
       - Box office: 1 per 150 expected attendees (ceil, min 1)
       - Concessions: 1 per 70 expected attendees (ceil, min 1)
       - Floor ushers: 1 per 120 expected attendees (ceil, min 1)
       - Cleaning crew: 1 per 100 auditorium seats (ceil, min 1)
       - Concession stock: 0.70 units per predicted attendee
    """
    cap = max(1.0, float(screen_capacity))
    upper = float(interval_80_upper)
    pred_tickets = float(predicted_tickets)
    occu = float(predicted_occupancy)

    # 1. Screen Allocation
    if occu >= 0.85 or upper >= cap:
        decision = "UPGRADE_SCREEN"
        action = f"High sell-out risk ({occu*100:.1f}% projected occupancy). Reallocate to a larger auditorium screen (350+ seats) to capture unmet demand."
        suggested_cap = "Large (300 - 450 seats)"
        basis = "Projected occupancy >= 85% or upper 80% confidence bound reaches hall capacity."
    elif occu <= 0.35 and cap > 180:
        decision = "DOWNGRADE_SCREEN"
        action = f"Auditorium under-utilized ({occu*100:.1f}% projected occupancy). Shift to a compact theater room (80-150 seats) to minimize HVAC & operational costs."
        suggested_cap = "Small (80 - 150 seats)"
        basis = "Projected occupancy <= 35% in large auditorium (> 180 seats)."
    else:
        decision = "MAINTAIN_CURRENT"
        action = f"Balanced demand ({occu*100:.1f}% projected occupancy). Maintain currently scheduled screening room."
        suggested_cap = "Current auditorium capacity is optimal."
        basis = "Occupancy within normal operational window (35% to 85%)."

    # 2. Staffing Rosters (Ceiling-based queuing formulas from upper bound)
    box_office = max(1, int(np.ceil(upper / 150.0)))
    concession = max(1, int(np.ceil(upper / 70.0)))
    ushers = max(1, int(np.ceil(upper / 120.0)))
    cleaning = max(1, int(np.ceil(cap / 100.0)))
    total_staff = box_office + concession + ushers + cleaning

    concession_units = int(round(pred_tickets * 0.70))

    staffing_details = StaffingBreakdown(
        box_office_staff=box_office,
        concession_cashiers=concession,
        floor_ushers=ushers,
        cleaning_crew=cleaning,
        total_floor_staff=total_staff,
        concession_inventory_units=concession_units,
        planning_basis="Risk-adjusted against upper 80% prediction bound to protect against asymmetric 1.5x under-staffing penalty.",
        role_ratios={
            "box_office": "1 employee per 150 guests (min 1)",
            "concession": "1 employee per 70 guests (min 1)",
            "floor_ushers": "1 employee per 120 guests (min 1)",
            "cleaning_crew": "1 employee per 100 auditorium seats (min 1)",
        }
    )

    screen_details = ScreenAllocationDecision(
        decision=decision,
        action=action,
        suggested_capacity_range=suggested_cap,
        occupancy_threshold_basis=basis,
    )

    return OperationalAdvisory(
        screen_decision=decision,
        recommended_action=action,
        suggested_capacity_range=suggested_cap,
        box_office_staff=box_office,
        ushers_count=ushers,
        concession_cashiers=concession,
        cleaning_crew=cleaning,
        total_floor_staff=total_staff,
        concession_inventory_units=concession_units,
        staffing_details=asdict(staffing_details),
        screen_details=asdict(screen_details),
        managerial_review_required=True,
        operational_disclaimer="Rule-based operational decision support only. Not ML output. Managerial sign-off required.",
    )
