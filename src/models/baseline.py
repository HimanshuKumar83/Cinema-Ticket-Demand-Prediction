"""
CinemaOps AI - Hierarchical Occupancy Baseline Model
Implements a 4-level hierarchical historical mean baseline with capacity adjustment.

Hierarchy levels:
  Level 1: (movie_id, cinema_id, show_slot)
  Level 2: (movie_id, show_slot)
  Level 3: (cinema_id, show_slot)
  Level 4: Global training occupancy mean

Multiplied by screen capacity and clipped to [0, capacity].
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


class HierarchicalOccupancyBaseline:
    """Deterministic, leakage-free hierarchical historical mean baseline."""

    def __init__(self):
        self.global_mean_occu: float = 0.50
        self.movie_cinema_slot_occu: Dict[tuple, float] = {}
        self.movie_slot_occu: Dict[tuple, float] = {}
        self.cinema_slot_occu: Dict[tuple, float] = {}
        self.is_fitted: bool = False

    def fit(
        self,
        df_train: pd.DataFrame,
        capacity_col: str = "screen_capacity",
        target_col: str = "actual_tickets_sold",
    ) -> "HierarchicalOccupancyBaseline":
        """Fit hierarchical lookups strictly on training split."""
        df = df_train.copy()
        
        # Support alias columns
        cap_col = capacity_col if capacity_col in df.columns else "capacity"
        tgt_col = target_col if target_col in df.columns else "tickets_sold"
        movie_col = "movie_id" if "movie_id" in df.columns else "film_code"
        cinema_col = "cinema_id" if "cinema_id" in df.columns else "cinema_code"
        slot_col = "show_slot" if "show_slot" in df.columns else "show_time"

        # Calculate training occupancy rate
        capacities = np.maximum(1.0, df[cap_col].to_numpy(dtype=float))
        occupancies = np.clip(df[tgt_col].to_numpy(dtype=float) / capacities, 0.0, 1.0)
        df["_occupancy"] = occupancies

        # Level 4: Global mean
        self.global_mean_occu = float(df["_occupancy"].mean())

        # Level 3: Cinema-Slot mean
        cs_grp = df.groupby([cinema_col, slot_col])["_occupancy"].mean()
        self.cinema_slot_occu = {k: float(v) for k, v in cs_grp.items()}

        # Level 2: Movie-Slot mean
        ms_grp = df.groupby([movie_col, slot_col])["_occupancy"].mean()
        self.movie_slot_occu = {k: float(v) for k, v in ms_grp.items()}

        # Level 1: Movie-Cinema-Slot mean
        mcs_grp = df.groupby([movie_col, cinema_col, slot_col])["_occupancy"].mean()
        self.movie_cinema_slot_occu = {k: float(v) for k, v in mcs_grp.items()}

        self.is_fitted = True
        return self

    def predict(
        self,
        df: pd.DataFrame,
        capacity_col: str = "screen_capacity",
    ) -> np.ndarray:
        """
        Generate predictions using 4-level fallback hierarchy:
        Level 1 -> Level 2 -> Level 3 -> Level 4 (Global)
        Clipped to [0, capacity].
        """
        if not self.is_fitted:
            raise RuntimeError("HierarchicalOccupancyBaseline must be fitted before predict.")

        df_eval = df.copy()
        cap_col = capacity_col if capacity_col in df_eval.columns else "capacity"
        movie_col = "movie_id" if "movie_id" in df_eval.columns else "film_code"
        cinema_col = "cinema_id" if "cinema_id" in df_eval.columns else "cinema_code"
        slot_col = "show_slot" if "show_slot" in df_eval.columns else "show_time"

        preds_occu = []
        for _, row in df_eval.iterrows():
            m_id = row.get(movie_col)
            c_id = row.get(cinema_col)
            s_val = row.get(slot_col)

            # Level 1: (movie, cinema, slot)
            val = self.movie_cinema_slot_occu.get((m_id, c_id, s_val))
            if val is None:
                # Level 2: (movie, slot)
                val = self.movie_slot_occu.get((m_id, s_val))
                if val is None:
                    # Level 3: (cinema, slot)
                    val = self.cinema_slot_occu.get((c_id, s_val))
                    if val is None:
                        # Level 4: Global mean
                        val = self.global_mean_occu

            preds_occu.append(val)

        pred_occu_arr = np.array(preds_occu, dtype=float)
        capacities = df_eval[cap_col].to_numpy(dtype=float)
        pred_tickets = np.round(pred_occu_arr * capacities, 1)

        # Enforce bounds strictly within [0, capacity]
        clipped_tickets = np.clip(pred_tickets, 0.0, capacities)
        return clipped_tickets
