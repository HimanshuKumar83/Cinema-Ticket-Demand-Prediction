"""Hierarchical Occupancy Baseline Model.
Levels of fallback:
1. Mean occupancy by (cinema_code, weekday, show_time) in training set
2. Mean occupancy by (weekday, show_time) in training set
3. Global mean occupancy in training set
Multiplied by screening capacity and clipped to [0, capacity].
"""
import numpy as np
import pandas as pd


class HierarchicalOccupancyBaseline:
    """Deterministic, leakage-free hierarchical historical mean baseline."""

    def __init__(self):
        self.global_mean_occu: float = 0.15
        self.slot_mean_occu: pd.Series = None
        self.cinema_slot_mean_occu: pd.Series = None

    def fit(self, df_train: pd.DataFrame, capacity_col: str = "capacity", target_col: str = "tickets_sold"):
        """Fit historical occupancy lookup on training split only."""
        train_df = df_train.copy()
        train_df["date"] = pd.to_datetime(train_df["date"])
        train_df["weekday"] = train_df["date"].dt.weekday
        
        # Calculate training occupancy
        train_df["occupancy"] = (train_df[target_col] / np.maximum(1, train_df[capacity_col])).clip(0.0, 1.0)
        
        # Level 3: Global mean
        self.global_mean_occu = float(train_df["occupancy"].mean())
        
        # Level 2: Slot mean by (weekday, show_time)
        self.slot_mean_occu = train_df.groupby(["weekday", "show_time"])["occupancy"].mean()
        
        # Level 1: Cinema-Slot mean by (cinema_code, weekday, show_time)
        self.cinema_slot_mean_occu = train_df.groupby(["cinema_code", "weekday", "show_time"])["occupancy"].mean()
        
        return self

    def predict(self, df: pd.DataFrame, capacity_col: str = "capacity") -> np.ndarray:
        """Generate ticket predictions clipped to [0, capacity]."""
        df_eval = df.copy()
        if "weekday" not in df_eval.columns:
            df_eval["weekday"] = pd.to_datetime(df_eval["date"]).dt.weekday
            
        preds_occu = []
        for _, row in df_eval.iterrows():
            c_code = row["cinema_code"]
            w_day = row["weekday"]
            s_time = row["show_time"]
            
            # Level 1 lookup
            val = self.cinema_slot_mean_occu.get((c_code, w_day, s_time), np.nan)
            if np.isnan(val):
                # Level 2 lookup
                val = self.slot_mean_occu.get((w_day, s_time), np.nan)
                if np.isnan(val):
                    # Level 3 lookup
                    val = self.global_mean_occu
                    
            preds_occu.append(val)
            
        pred_occu_arr = np.array(preds_occu)
        capacities = df_eval[capacity_col].to_numpy(dtype=float)
        pred_tickets = pred_occu_arr * capacities
        
        # Clip to [0, capacity]
        clipped_tickets = np.clip(pred_tickets, 0.0, capacities)
        return clipped_tickets
