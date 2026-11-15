"""Data cleaning and leakage table generation for Cinema Tickets dataset."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

RAW_DATA_PATH = Path("data/raw/cinemaTicket_Ref.csv")
REPORTS_DIR = Path("reports")
PROCESSED_DIR = Path("data/processed")


def generate_leakage_table() -> pd.DataFrame:
    """Generate and save data leakage audit table."""
    leakage_records = [
        {
            "feature": "tickets_sold",
            "available_at_t_minus_48h": False,
            "category": "Target Variable",
            "action": "TARGET (y)",
            "reason": "Primary prediction target representing show attendance."
        },
        {
            "feature": "occu_perc",
            "available_at_t_minus_48h": False,
            "category": "Target-Derived",
            "action": "DROP",
            "reason": "Direct mathematical transformation of target: (tickets_sold / capacity) * 100."
        },
        {
            "feature": "total_sales",
            "available_at_t_minus_48h": False,
            "category": "Post-Show Outcome",
            "action": "DROP",
            "reason": "Total revenue recorded after ticket transactions; contemporaneous with target."
        },
        {
            "feature": "tickets_out",
            "available_at_t_minus_48h": False,
            "category": "Post-Show Outcome",
            "action": "DROP",
            "reason": "Cancelled/refunded tickets recorded after the booking window."
        },
        {
            "feature": "ticket_use",
            "available_at_t_minus_48h": False,
            "category": "Post-Show Outcome",
            "action": "DROP",
            "reason": "Turnstile/admittance count verified at the theater door."
        },
        {
            "feature": "ticket_price",
            "available_at_t_minus_48h": False,
            "category": "Contemporaneous Ratio",
            "action": "DROP_CONTEMPORANEOUS",
            "reason": "Exact ex-post quotient (total_sales / tickets_sold). Dropped to eliminate leak. Only past-lagged price allowed."
        },
        {
            "feature": "capacity",
            "available_at_t_minus_48h": True,
            "category": "Physical Theater Attribute",
            "action": "KEEP (with imputation)",
            "reason": "Seat count of the screening auditorium is fixed and known before showtime."
        },
        {
            "feature": "show_time",
            "available_at_t_minus_48h": True,
            "category": "Schedule Metadata",
            "action": "KEEP",
            "reason": "Screening slot index published in the cinema timetable."
        },
        {
            "feature": "cinema_code",
            "available_at_t_minus_48h": True,
            "category": "Identifier",
            "action": "KEEP",
            "reason": "Cinema complex identifier known at schedule time."
        },
        {
            "feature": "film_code",
            "available_at_t_minus_48h": True,
            "category": "Identifier",
            "action": "KEEP",
            "reason": "Movie scheduled for the screening slot."
        },
        {
            "feature": "date",
            "available_at_t_minus_48h": True,
            "category": "Calendar Metadata",
            "action": "KEEP",
            "reason": "Scheduled screening date used for temporal indexing and calendar feature extraction."
        }
    ]

    df_leakage = pd.DataFrame(leakage_records)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df_leakage.to_csv(REPORTS_DIR / "leakage_table.csv", index=False)
    
    # Save markdown version without external tabulate dependency
    headers = list(df_leakage.columns)
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    row_lines = []
    for _, row in df_leakage.iterrows():
        row_lines.append("| " + " | ".join(str(val) for val in row.values) + " |")
    
    md_content = "# Data Leakage Audit Table\n\n"
    md_content += "Assessment of features available at $T-48\\text{h}$ prediction time vs. ex-post post-show leakage.\n\n"
    md_content += "\n".join([header_line, separator_line] + row_lines) + "\n"
    with open(REPORTS_DIR / "leakage_table.md", "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(f"Leakage table saved to {REPORTS_DIR / 'leakage_table.csv'} and .md")
    return df_leakage


def clean_dataset(csv_path: Path = RAW_DATA_PATH) -> tuple[pd.DataFrame, dict]:
    """Clean the raw cinema dataset according to strict operational rules."""
    df_raw = pd.read_csv(csv_path)
    raw_row_count = len(df_raw)
    
    # Rename target column if needed
    if "tickets_num" in df_raw.columns:
        df_raw = df_raw.rename(columns={"tickets_num": "tickets_sold"})
        
    # Ensure datetime
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    
    # Cleaning rule 1: Aggregate composite key duplicates on (cinema_code, film_code, date, show_time)
    key_cols = ["cinema_code", "film_code", "date", "show_time"]
    dup_rows_count = df_raw.duplicated(subset=key_cols, keep=False).sum()
    
    # Aggregation mapping
    agg_dict = {
        "tickets_sold": "sum",
        "total_sales": "sum",
        "capacity": "max",
        "tickets_out": "sum",
        "ticket_use": "sum",
        "ticket_price": "mean",
        "occu_perc": "mean"
    }
    # Add non-numeric or calendar columns
    for col in df_raw.columns:
        if col not in key_cols and col not in agg_dict:
            agg_dict[col] = "first"
            
    df_agg = df_raw.groupby(key_cols, as_index=False).agg(agg_dict)
    dedup_row_count = len(df_agg)
    rows_collapsed = raw_row_count - dedup_row_count

    # Cleaning rule 2: Capacity imputation
    # Compute median valid capacity by cinema
    valid_cap_mask = df_agg["capacity"] > 0
    cinema_median_caps = df_agg[valid_cap_mask].groupby("cinema_code")["capacity"].median()
    global_median_cap = float(df_agg[valid_cap_mask]["capacity"].median())

    df_agg["capacity_imputed"] = 0
    invalid_cap_mask = (~valid_cap_mask) | df_agg["capacity"].isna()
    imputed_count = int(invalid_cap_mask.sum())

    if imputed_count > 0:
        imputed_caps = df_agg.loc[invalid_cap_mask, "cinema_code"].map(cinema_median_caps).fillna(global_median_cap)
        df_agg.loc[invalid_cap_mask, "capacity"] = imputed_caps
        df_agg.loc[invalid_cap_mask, "capacity_imputed"] = 1

    # Cleaning rule 3: Overcapacity sanitization
    # When tickets_sold > capacity (0.11% of rows), clip tickets to capacity so occupancy <= 1.0
    overcap_mask = df_agg["tickets_sold"] > df_agg["capacity"]
    overcap_count = int(overcap_mask.sum())
    df_agg.loc[overcap_mask, "tickets_sold"] = df_agg.loc[overcap_mask, "capacity"].round().astype(int)

    # Cleaning rule 4: Compute sanitized occupancy
    df_agg["occupancy"] = (df_agg["tickets_sold"] / df_agg["capacity"]).clip(0.0, 1.0)

    # Sort chronologically
    df_agg = df_agg.sort_values(by=["date", "show_time", "cinema_code", "film_code"]).reset_index(drop=True)

    cleaning_report = {
        "raw_rows": raw_row_count,
        "duplicate_rows_in_key": int(dup_rows_count),
        "rows_after_key_aggregation": dedup_row_count,
        "rows_collapsed": int(rows_collapsed),
        "capacity_imputed_rows": imputed_count,
        "global_median_capacity": global_median_cap,
        "overcapacity_shows_clipped": overcap_count,
        "final_clean_rows": len(df_agg)
    }

    # Save cleaning report
    with open(REPORTS_DIR / "cleaning_report.json", "w", encoding="utf-8") as f:
        json.dump(cleaning_report, f, indent=2)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_agg.to_parquet(PROCESSED_DIR / "cleaned_cinema_shows.parquet", index=False)
    print(f"Cleaned dataset saved: {len(df_agg):,} rows to {PROCESSED_DIR / 'cleaned_cinema_shows.parquet'}")
    print(f"Cleaning stats: {cleaning_report}")

    return df_agg, cleaning_report


if __name__ == "__main__":
    generate_leakage_table()
    clean_dataset()
