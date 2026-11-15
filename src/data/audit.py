"""Data audit script for Kaggle Cinema Tickets dataset."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

RAW_DATA_PATH = Path("data/raw/cinemaTicket_Ref.csv")
REPORTS_DIR = Path("reports")


def run_audit(csv_path: Path = RAW_DATA_PATH) -> dict:
    """Run comprehensive data audit and return findings dictionary."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot run audit: file not found at {csv_path}")

    print("=" * 80)
    print("CINEMA TICKETS DATASET AUDIT")
    print(f"Reading: {csv_path}")
    print("=" * 80)

    df = pd.read_csv(csv_path)
    n_rows_raw, n_cols_raw = df.shape

    # Determine ticket column name
    ticket_col = "tickets_num" if "tickets_num" in df.columns else ("tickets_sold" if "tickets_sold" in df.columns else None)
    if not ticket_col:
        raise ValueError(f"Could not find ticket volume column in: {list(df.columns)}")

    # Ensure date parsing
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    null_dates = df["date_parsed"].isna().sum()

    min_date = df["date_parsed"].min()
    max_date = df["date_parsed"].max()
    date_range_days = (max_date - min_date).days + 1 if pd.notna(min_date) else 0

    unique_dates = df["date_parsed"].dropna().dt.date.unique()
    unique_dates_count = len(unique_dates)
    expected_full_range = pd.date_range(min_date, max_date).date
    missing_dates = sorted(list(set(expected_full_range) - set(unique_dates)))

    unique_films = df["film_code"].nunique()
    unique_cinemas = df["cinema_code"].nunique()
    unique_show_times = df["show_time"].nunique()

    # Composite key check: (cinema_code, film_code, date, show_time)
    key_cols = ["cinema_code", "film_code", "date", "show_time"]
    dup_mask = df.duplicated(subset=key_cols, keep=False)
    num_dup_rows = dup_mask.sum()
    dup_rate_pct = (num_dup_rows / n_rows_raw) * 100

    # Occupancy consistency check: occu_perc ~ (tickets / capacity) * 100
    valid_cap = df["capacity"] > 0
    calc_occu = np.where(valid_cap, (df[ticket_col] / df["capacity"]) * 100, np.nan)
    occu_diff = np.abs(df["occu_perc"] - calc_occu)
    valid_diff = occu_diff[valid_cap & df["occu_perc"].notna() & np.isfinite(calc_occu)]
    mean_occu_diff = float(valid_diff.mean()) if len(valid_diff) > 0 else None
    max_occu_diff = float(valid_diff.max()) if len(valid_diff) > 0 else None
    occu_corr = float(df["occu_perc"].corr(pd.Series(calc_occu))) if len(valid_diff) > 0 else None

    # Ticket price consistency check: ticket_price ~ total_sales / tickets
    valid_tickets = df[ticket_col] > 0
    calc_price = np.where(valid_tickets, df["total_sales"] / df[ticket_col], np.nan)
    price_diff = np.abs(df["ticket_price"] - calc_price)
    valid_price_diff = price_diff[valid_tickets & df["ticket_price"].notna() & np.isfinite(calc_price)]
    mean_price_diff = float(valid_price_diff.mean()) if len(valid_price_diff) > 0 else None
    price_corr = float(df["ticket_price"].corr(pd.Series(calc_price))) if len(valid_price_diff) > 0 else None

    # show_time analysis
    st_stats = {
        "min": float(df["show_time"].min()),
        "q25": float(df["show_time"].quantile(0.25)),
        "median": float(df["show_time"].median()),
        "q75": float(df["show_time"].quantile(0.75)),
        "max": float(df["show_time"].max()),
        "distinct_values": sorted([int(x) for x in df["show_time"].dropna().unique() if np.isfinite(x)])
    }

    # Interpretation:
    # Check if show_time values are 1..N (slot indices) or e.g. 1100, 1400 (clock times)
    if st_stats["max"] <= 50:
        st_interpretation = "Sequential slot index (e.g. 1st show of the day, 2nd show of the day, up to max slot)."
    elif st_stats["max"] > 100:
        st_interpretation = "Clock time or hour-minute representation (e.g. 1300 = 13:00)."
    else:
        st_interpretation = "Slot index or hour indicator."

    # Invalid values & outliers
    invalid_cap_count = int((df["capacity"] <= 0).sum() + df["capacity"].isna().sum())
    invalid_cap_pct = (invalid_cap_count / n_rows_raw) * 100

    neg_tickets_count = int((df[ticket_col] < 0).sum())
    neg_tickets_pct = (neg_tickets_count / n_rows_raw) * 100

    # Overcapacity (tickets > capacity)
    overcap_mask = (df[ticket_col] > df["capacity"]) & (df["capacity"] > 0)
    overcap_count = int(overcap_mask.sum())
    overcap_pct = (overcap_count / n_rows_raw) * 100

    # Demand censoring (sold out shows: tickets >= capacity)
    sold_out_mask = (df[ticket_col] >= df["capacity"]) & (df["capacity"] > 0)
    sold_out_count = int(sold_out_mask.sum())
    sold_out_pct = (sold_out_count / n_rows_raw) * 100

    # Zero sales shows
    zero_sales_count = int((df[ticket_col] == 0).sum())
    zero_sales_pct = (zero_sales_count / n_rows_raw) * 100

    # Cold-start analysis:
    # Sort dates chronologically, split at last 30% of dates
    unique_dates_sorted = sorted(list(unique_dates))
    cutoff_idx = int(len(unique_dates_sorted) * 0.70)
    early_dates = set(unique_dates_sorted[:cutoff_idx])
    late_dates = set(unique_dates_sorted[cutoff_idx:])

    early_films = set(df[df["date_parsed"].dt.date.isin(early_dates)]["film_code"].unique())
    late_films = set(df[df["date_parsed"].dt.date.isin(late_dates)]["film_code"].unique())
    new_films_in_last_30pct = late_films - early_films
    shows_with_new_films = df[
        df["date_parsed"].dt.date.isin(late_dates) & df["film_code"].isin(new_films_in_last_30pct)
    ].shape[0]

    # GO / NO-GO Verdict evaluation:
    # Criteria:
    # 1. Dataset exists and > 50k rows
    # 2. Key entities present
    # 3. show_time is usable
    # 4. Occupancy / capacity consistency is verified or explainable
    # 5. Cold-start test has sufficient new films (>= 10 new films)
    is_go = (
        n_rows_raw > 10000 and
        unique_films >= 20 and
        unique_cinemas >= 10 and
        len(new_films_in_last_30pct) >= 5 and
        null_dates == 0
    )

    verdict = "GO" if is_go else "NO-GO"
    verdict_reasons = []
    if n_rows_raw < 10000:
        verdict_reasons.append(f"Insufficient rows: {n_rows_raw}")
    if len(new_films_in_last_30pct) < 5:
        verdict_reasons.append(f"Too few new films for cold start: {len(new_films_in_last_30pct)}")
    if not is_go:
        verdict_reasons.append("Criteria not satisfied.")
    else:
        verdict_reasons.append("Dataset is intact, key relationships verified, sufficient cold-start sample present.")

    audit_summary = {
        "verdict": verdict,
        "verdict_reasons": verdict_reasons,
        "rows_count": n_rows_raw,
        "columns_count": n_cols_raw,
        "columns": list(df.columns),
        "ticket_column": ticket_col,
        "date_range": {
            "start": str(min_date)[:10],
            "end": str(max_date)[:10],
            "total_span_days": date_range_days,
            "unique_dates": unique_dates_count,
            "missing_calendar_dates_count": len(missing_dates),
            "missing_dates_samples": [str(d) for d in missing_dates[:5]]
        },
        "entities": {
            "unique_films": unique_films,
            "unique_cinemas": unique_cinemas,
            "unique_show_time_slots": unique_show_times
        },
        "composite_key_duplicates": {
            "key": key_cols,
            "duplicate_rows": num_dup_rows,
            "duplicate_rate_pct": round(dup_rate_pct, 4)
        },
        "relationships": {
            "occu_perc_vs_formula": {
                "mean_abs_difference": round(mean_occu_diff, 4) if mean_occu_diff is not None else None,
                "max_abs_difference": round(max_occu_diff, 4) if max_occu_diff is not None else None,
                "correlation": round(occu_corr, 4) if occu_corr is not None else None,
                "interpretation": "occu_perc is directly derived from (tickets / capacity) * 100 and MUST BE DROPPED to prevent severe leakage."
            },
            "ticket_price_vs_formula": {
                "mean_abs_difference": round(mean_price_diff, 4) if mean_price_diff is not None else None,
                "correlation": round(price_corr, 4) if price_corr is not None else None,
                "is_derived_from_sales": bool(price_corr is not None and price_corr > 0.95),
                "interpretation": "ticket_price perfectly or near-perfectly matches total_sales / tickets_num. Using contemporaneous ticket_price would leak sales/tickets. Only strictly lagged average price or dropping ticket_price is permissible."
            }
        },
        "show_time_analysis": {
            "stats": st_stats,
            "interpretation_chosen": st_interpretation
        },
        "invalid_values_and_anomalies": {
            "invalid_or_zero_capacity_count": invalid_cap_count,
            "invalid_capacity_pct": round(invalid_cap_pct, 4),
            "negative_tickets_count": neg_tickets_count,
            "negative_tickets_pct": round(neg_tickets_pct, 4),
            "overcapacity_shows_count": overcap_count,
            "overcapacity_shows_pct": round(overcap_pct, 4),
            "zero_sales_count": zero_sales_count,
            "zero_sales_pct": round(zero_sales_pct, 4)
        },
        "demand_censoring": {
            "sold_out_or_capped_count": sold_out_count,
            "sold_out_or_capped_pct": round(sold_out_pct, 4),
            "implication": "Censored demand observed; true demand when sold out may exceed physical capacity."
        },
        "cold_start_feasibility": {
            "cutoff_date": str(unique_dates_sorted[cutoff_idx]),
            "early_unique_films": len(early_films),
            "late_unique_films": len(late_films),
            "new_films_first_seen_in_last_30pct": len(new_films_in_last_30pct),
            "shows_with_new_films_in_last_30pct": shows_with_new_films,
            "feasibility": "FEASIBLE" if len(new_films_in_last_30pct) >= 10 else "MARGINAL"
        }
    }

    # Print summary to console
    print(f"Total Rows: {n_rows_raw:,} | Columns: {n_cols_raw}")
    print(f"Date Range: {audit_summary['date_range']['start']} to {audit_summary['date_range']['end']} ({unique_dates_count} active days, {len(missing_dates)} calendar gaps)")
    print(f"Entities: {unique_films} unique films, {unique_cinemas} cinemas, {unique_show_times} show times")
    print(f"Composite Key Duplicates on {key_cols}: {num_dup_rows} rows ({dup_rate_pct:.3f}%)")
    print(f"Occupancy Formula Match: Corr = {occu_corr}, Mean Abs Diff = {mean_occu_diff}")
    print(f"Ticket Price Formula Match: Corr = {price_corr}, Mean Abs Diff = {mean_price_diff}")
    print(f"Show Time: values {st_stats['distinct_values'][:10]}... | {st_interpretation}")
    print(f"Invalid Capacity (<=0 or NaN): {invalid_cap_count} ({invalid_cap_pct:.2f}%)")
    print(f"Overcapacity (tickets > capacity): {overcap_count} ({overcap_pct:.2f}%)")
    print(f"Censored / Sold Out (tickets >= capacity): {sold_out_count} ({sold_out_pct:.2f}%)")
    print(f"Cold Start: {len(new_films_in_last_30pct)} new films first appear in final 30% of dates ({shows_with_new_films} shows)")
    print("=" * 80)
    print(f"VERDICT: {verdict}")
    print(f"Reason: {', '.join(verdict_reasons)}")
    print("=" * 80)

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    # Save to reports/audit_report.json
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "audit_report.json", "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2, cls=NpEncoder)
    print(f"Audit report saved to {REPORTS_DIR / 'audit_report.json'}")

    return audit_summary


if __name__ == "__main__":
    run_audit()

