"""Feature engineering pipeline for Cinema Ticket Demand Forecasting.
Computes strictly past-only features shifted by >= 2 days (T-48h horizon).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ARTIFACTS_DIR = Path("artifacts")
PROCESSED_DIR = Path("data/processed")


class HistoryFeatureStore:
    """Stores cleaned historical show records and computes leakage-free T-48h features.
    Strict horizon: For any show on date D, only data from date <= D - 2 days is accessible.
    """

    def __init__(self, history_df: pd.DataFrame):
        self.history_df = history_df.copy()
        self.history_df["date"] = pd.to_datetime(self.history_df["date"])
        
        # Precompute daily summaries for high performance
        # 1. Film daily
        self.film_daily = (
            self.history_df.groupby(["film_code", "date"])
            .agg(
                daily_tickets=("tickets_sold", "sum"),
                daily_shows=("tickets_sold", "count"),
                daily_occupancy=("occupancy", "mean")
            )
            .reset_index()
        )
        
        # 2. Cinema daily
        self.cinema_daily = (
            self.history_df.groupby(["cinema_code", "date"])
            .agg(
                daily_tickets=("tickets_sold", "sum"),
                daily_shows=("tickets_sold", "count"),
                daily_occupancy=("occupancy", "mean")
            )
            .reset_index()
        )

        # 3. Cinema-Slot daily
        self.cinema_slot_daily = (
            self.history_df.groupby(["cinema_code", "show_time", "date"])
            .agg(
                daily_tickets=("tickets_sold", "sum"),
                daily_shows=("tickets_sold", "count"),
                daily_occupancy=("occupancy", "mean")
            )
            .reset_index()
        )

        # 4. Film-Cinema daily
        self.film_cinema_daily = (
            self.history_df.groupby(["film_code", "cinema_code", "date"])
            .agg(
                daily_tickets=("tickets_sold", "sum"),
                daily_shows=("tickets_sold", "count"),
                daily_occupancy=("occupancy", "mean")
            )
            .reset_index()
        )

        # Precompute film first appearance date across the dataset
        self.film_first_seen = self.history_df.groupby("film_code")["date"].min().to_dict()
        self.cinema_first_seen = self.history_df.groupby("cinema_code")["date"].min().to_dict()

    @classmethod
    def from_parquet(cls, parquet_path: Path):
        df = pd.read_parquet(parquet_path)
        return cls(df)

    def save_lookup(self, save_path: Path):
        """Save history lookup dataframe for inference."""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cols_to_keep = [
            "date", "cinema_code", "film_code", "show_time", 
            "tickets_sold", "capacity", "occupancy"
        ]
        self.history_df[cols_to_keep].to_parquet(save_path, index=False)
        print(f"History lookup saved to {save_path} ({len(self.history_df):,} rows)")


def compute_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute deterministic calendar and capacity features known at schedule time."""
    dates = pd.to_datetime(df["date"])
    out = pd.DataFrame(index=df.index)
    
    out["weekday"] = dates.dt.weekday
    out["month"] = dates.dt.month
    out["quarter"] = dates.dt.quarter
    out["week_of_year"] = dates.dt.isocalendar().week.astype(int)
    out["is_weekend"] = (dates.dt.weekday >= 4).astype(int)  # Fri, Sat, Sun
    out["show_time"] = df["show_time"].astype(int)
    
    # Capacity features
    out["capacity"] = df["capacity"].astype(float)
    out["log_capacity"] = np.log1p(np.maximum(0, df["capacity"]))
    out["capacity_imputed"] = df.get("capacity_imputed", 0).astype(int)
    
    return out


def extract_features_for_dataset(df_target: pd.DataFrame, history_store: HistoryFeatureStore) -> pd.DataFrame:
    """Compute full feature matrix for a target dataframe using history_store.
    Enforces horizon: show on date D only sees history <= D - 2 days.
    """
    df_target = df_target.copy()
    df_target["date"] = pd.to_datetime(df_target["date"])
    
    # Base calendar and capacity
    feats = compute_calendar_features(df_target)
    feats["cinema_code"] = df_target["cinema_code"].astype(int)
    feats["film_code"] = df_target["film_code"].astype(int)
    feats["date"] = df_target["date"]

    # We compute rolling historical statistics by iterating through unique target dates.
    # Because there are only 234 unique dates, date-level batching is extremely fast (< 3 seconds)
    unique_dates = df_target["date"].unique()
    
    history_records = []
    
    for target_date in unique_dates:
        t_date = pd.Timestamp(target_date)
        # Prediction horizon cutoff: T-48h means strictly <= t_date - 2 days
        cutoff_date = t_date - pd.Timedelta(days=2)
        
        # Windows:
        # 7-day window: [t_date - 8 days, cutoff_date]
        w7_start = t_date - pd.Timedelta(days=8)
        # 14-day window: [t_date - 15 days, cutoff_date]
        w14_start = t_date - pd.Timedelta(days=15)
        # Same weekday last week: t_date - 7 days
        same_weekday_date = t_date - pd.Timedelta(days=7)

        # Slice daily history tables
        # 1. Film
        f_hist_all = history_store.film_daily[history_store.film_daily["date"] <= cutoff_date]
        f_7d = f_hist_all[f_hist_all["date"] >= w7_start].groupby("film_code").agg(
            f_tickets_7d=("daily_tickets", "sum"),
            f_shows_7d=("daily_shows", "sum"),
            f_occu_7d=("daily_occupancy", "mean")
        )
        f_7d["film_mean_tickets_7d"] = f_7d["f_tickets_7d"] / np.maximum(1, f_7d["f_shows_7d"])
        f_7d["film_mean_occu_7d"] = f_7d["f_occu_7d"]

        f_14d = f_hist_all[f_hist_all["date"] >= w14_start].groupby("film_code").agg(
            f_tickets_14d=("daily_tickets", "sum"),
            f_shows_14d=("daily_shows", "sum"),
            f_occu_14d=("daily_occupancy", "mean")
        )
        f_14d["film_mean_tickets_14d"] = f_14d["f_tickets_14d"] / np.maximum(1, f_14d["f_shows_14d"])
        f_14d["film_mean_occu_14d"] = f_14d["f_occu_14d"]

        f_lifetime = f_hist_all.groupby("film_code").agg(
            film_history_count=("daily_shows", "sum"),
            film_first_date=("date", "min")
        )

        f_same_weekday = f_hist_all[f_hist_all["date"] == same_weekday_date].set_index("film_code")[
            "daily_tickets"
        ].rename("film_tickets_same_weekday_last_week")

        # 2. Cinema
        c_hist_all = history_store.cinema_daily[history_store.cinema_daily["date"] <= cutoff_date]
        c_7d = c_hist_all[c_hist_all["date"] >= w7_start].groupby("cinema_code").agg(
            c_tickets_7d=("daily_tickets", "sum"),
            c_shows_7d=("daily_shows", "sum"),
            c_occu_7d=("daily_occupancy", "mean")
        )
        c_7d["cinema_mean_tickets_7d"] = c_7d["c_tickets_7d"] / np.maximum(1, c_7d["c_shows_7d"])
        c_7d["cinema_mean_occu_7d"] = c_7d["c_occu_7d"]

        c_14d = c_hist_all[c_hist_all["date"] >= w14_start].groupby("cinema_code").agg(
            c_tickets_14d=("daily_tickets", "sum"),
            c_shows_14d=("daily_shows", "sum"),
            c_occu_14d=("daily_occupancy", "mean")
        )
        c_14d["cinema_mean_tickets_14d"] = c_14d["c_tickets_14d"] / np.maximum(1, c_14d["c_shows_14d"])
        c_14d["cinema_mean_occu_14d"] = c_14d["c_occu_14d"]

        c_lifetime = c_hist_all.groupby("cinema_code").agg(
            cinema_history_count=("daily_shows", "sum")
        )

        c_same_weekday = c_hist_all[c_hist_all["date"] == same_weekday_date].set_index("cinema_code")[
            "daily_occupancy"
        ].rename("cinema_occu_same_weekday_last_week")

        # 3. Cinema-Slot
        cs_hist_all = history_store.cinema_slot_daily[history_store.cinema_slot_daily["date"] <= cutoff_date]
        cs_14d = cs_hist_all[cs_hist_all["date"] >= w14_start].groupby(["cinema_code", "show_time"]).agg(
            cs_tickets_14d=("daily_tickets", "sum"),
            cs_shows_14d=("daily_shows", "sum"),
            cs_occu_14d=("daily_occupancy", "mean")
        )
        cs_14d["cinema_slot_mean_tickets_14d"] = cs_14d["cs_tickets_14d"] / np.maximum(1, cs_14d["cs_shows_14d"])
        cs_14d["cinema_slot_mean_occu_14d"] = cs_14d["cs_occu_14d"]

        # 4. Film-Cinema
        fc_hist_all = history_store.film_cinema_daily[history_store.film_cinema_daily["date"] <= cutoff_date]
        fc_14d = fc_hist_all[fc_hist_all["date"] >= w14_start].groupby(["film_code", "cinema_code"]).agg(
            fc_tickets_14d=("daily_tickets", "sum"),
            fc_shows_14d=("daily_shows", "sum"),
            fc_occu_14d=("daily_occupancy", "mean")
        )
        fc_14d["film_cinema_mean_tickets_14d"] = fc_14d["fc_tickets_14d"] / np.maximum(1, fc_14d["fc_shows_14d"])
        fc_14d["film_cinema_mean_occu_14d"] = fc_14d["fc_occu_14d"]

        # Build rows for this target_date
        sub_df = df_target[df_target["date"] == t_date][["film_code", "cinema_code", "show_time"]].copy()
        
        # Film mappings
        sub_df["film_mean_tickets_7d"] = sub_df["film_code"].map(f_7d["film_mean_tickets_7d"])
        sub_df["film_mean_occu_7d"] = sub_df["film_code"].map(f_7d["film_mean_occu_7d"])
        sub_df["film_mean_tickets_14d"] = sub_df["film_code"].map(f_14d["film_mean_tickets_14d"])
        sub_df["film_mean_occu_14d"] = sub_df["film_code"].map(f_14d["film_mean_occu_14d"])
        sub_df["film_history_count"] = sub_df["film_code"].map(f_lifetime["film_history_count"]).fillna(0)
        
        film_first_dates = sub_df["film_code"].map(f_lifetime["film_first_date"])
        sub_df["days_since_first_show"] = np.where(
            film_first_dates.notna(),
            (t_date - film_first_dates).dt.days,
            -1
        )
        sub_df["is_opening_week"] = (
            (sub_df["days_since_first_show"] >= 0) & (sub_df["days_since_first_show"] <= 7)
        ).astype(int)
        sub_df["film_tickets_same_weekday_last_week"] = sub_df["film_code"].map(f_same_weekday)

        # Cinema mappings
        sub_df["cinema_mean_tickets_7d"] = sub_df["cinema_code"].map(c_7d["cinema_mean_tickets_7d"])
        sub_df["cinema_mean_occu_7d"] = sub_df["cinema_code"].map(c_7d["cinema_mean_occu_7d"])
        sub_df["cinema_mean_tickets_14d"] = sub_df["cinema_code"].map(c_14d["cinema_mean_tickets_14d"])
        sub_df["cinema_mean_occu_14d"] = sub_df["cinema_code"].map(c_14d["cinema_mean_occu_14d"])
        sub_df["cinema_history_count"] = sub_df["cinema_code"].map(c_lifetime["cinema_history_count"]).fillna(0)
        sub_df["cinema_occu_same_weekday_last_week"] = sub_df["cinema_code"].map(c_same_weekday)

        # Cinema-Slot mappings
        cs_keys = list(zip(sub_df["cinema_code"], sub_df["show_time"]))
        sub_df["cinema_slot_mean_tickets_14d"] = pd.Series(
            [cs_14d["cinema_slot_mean_tickets_14d"].get(k, np.nan) for k in cs_keys],
            index=sub_df.index
        )
        sub_df["cinema_slot_mean_occu_14d"] = pd.Series(
            [cs_14d["cinema_slot_mean_occu_14d"].get(k, np.nan) for k in cs_keys],
            index=sub_df.index
        )

        # Film-Cinema mappings
        fc_keys = list(zip(sub_df["film_code"], sub_df["cinema_code"]))
        sub_df["film_cinema_mean_tickets_14d"] = pd.Series(
            [fc_14d["film_cinema_mean_tickets_14d"].get(k, np.nan) for k in fc_keys],
            index=sub_df.index
        )
        sub_df["film_cinema_mean_occu_14d"] = pd.Series(
            [fc_14d["film_cinema_mean_occu_14d"].get(k, np.nan) for k in fc_keys],
            index=sub_df.index
        )

        history_records.append(sub_df)

    history_feats = pd.concat(history_records).loc[df_target.index]
    
    # Merge engineered features
    cols_to_add = [c for c in history_feats.columns if c not in ["film_code", "cinema_code", "show_time"]]
    for col in cols_to_add:
        feats[col] = history_feats[col]

    # Cold-start indicator flags
    feats["is_new_film"] = (feats["film_history_count"] == 0).astype(int)
    feats["is_new_cinema"] = (feats["cinema_history_count"] == 0).astype(int)
    feats["is_low_history"] = (
        (feats["film_history_count"] < 10) | (feats["cinema_history_count"] < 10)
    ).astype(int)

    # Missing value imputation: fill NaN in rolling features with fallback priors
    # For film rolling features: fallback to cinema mean or global median
    feats["film_mean_occu_7d_filled"] = feats["film_mean_occu_7d"].fillna(feats["cinema_mean_occu_7d"]).fillna(0.15)
    feats["cinema_mean_occu_7d_filled"] = feats["cinema_mean_occu_7d"].fillna(0.15)
    feats["film_mean_tickets_7d_filled"] = feats["film_mean_tickets_7d"].fillna(feats["cinema_mean_tickets_7d"]).fillna(30.0)
    feats["cinema_mean_tickets_7d_filled"] = feats["cinema_mean_tickets_7d"].fillna(30.0)

    # Impute remaining numeric NaN with -1 or sensible fallback for tree models
    numeric_cols = feats.select_dtypes(include=[np.number]).columns
    feats[numeric_cols] = feats[numeric_cols].fillna(-1.0)

    return feats


def create_chronological_splits(df_cleaned: pd.DataFrame) -> dict:
    """Create ~70% train / ~15% validation / ~15% test chronological split."""
    dates = sorted(df_cleaned["date"].unique())
    n = len(dates)
    train_end_idx = int(n * 0.70)
    val_end_idx = int(n * 0.85)

    train_dates = dates[:train_end_idx]
    val_dates = dates[train_end_idx:val_end_idx]
    test_dates = dates[val_end_idx:]

    df_train = df_cleaned[df_cleaned["date"].isin(train_dates)].copy()
    df_val = df_cleaned[df_cleaned["date"].isin(val_dates)].copy()
    df_test = df_cleaned[df_cleaned["date"].isin(test_dates)].copy()

    # Identify cold-start subsets in Test
    train_films = set(df_train["film_code"].unique())
    val_films = set(df_val["film_code"].unique())
    
    # New film subset: films never seen in training
    is_new_film_test = df_test["film_code"].isin(set(df_test["film_code"].unique()) - train_films)
    df_test_new_films = df_test[is_new_film_test].copy()

    # Low history subset in Test: film has < 10 shows in history at test time
    # (computed dynamically from history)

    split_info = {
        "train": {
            "start_date": str(pd.Timestamp(train_dates[0]))[:10],
            "end_date": str(pd.Timestamp(train_dates[-1]))[:10],
            "days_count": len(train_dates),
            "rows_count": len(df_train),
            "unique_films": int(df_train["film_code"].nunique()),
            "unique_cinemas": int(df_train["cinema_code"].nunique())
        },
        "validation": {
            "start_date": str(pd.Timestamp(val_dates[0]))[:10],
            "end_date": str(pd.Timestamp(val_dates[-1]))[:10],
            "days_count": len(val_dates),
            "rows_count": len(df_val),
            "unique_films": int(df_val["film_code"].nunique()),
            "new_films_vs_train": int(len(val_films - train_films))
        },
        "test": {
            "start_date": str(pd.Timestamp(test_dates[0]))[:10],
            "end_date": str(pd.Timestamp(test_dates[-1]))[:10],
            "days_count": len(test_dates),
            "rows_count": len(df_test),
            "unique_films": int(df_test["film_code"].nunique()),
            "new_films_vs_train": int(len(set(df_test["film_code"].unique()) - train_films)),
            "new_films_shows_count": len(df_test_new_films)
        }
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_DIR / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)

    print(f"Split info saved to {ARTIFACTS_DIR / 'split_info.json'}")
    return {
        "train": df_train,
        "val": df_val,
        "test": df_test,
        "split_info": split_info
    }


if __name__ == "__main__":
    df_clean = pd.read_parquet(PROCESSED_DIR / "cleaned_cinema_shows.parquet")
    splits = create_chronological_splits(df_clean)
    
    # Test feature extraction on a small slice
    history_store = HistoryFeatureStore(df_clean)
    history_store.save_lookup(ARTIFACTS_DIR / "history.parquet")
    
    print("Testing feature extraction on validation split...")
    val_sample = splits["val"].head(100)
    feats = extract_features_for_dataset(val_sample, history_store)
    print(f"Engineered {feats.shape[1]} features successfully:")
    print(list(feats.columns))

