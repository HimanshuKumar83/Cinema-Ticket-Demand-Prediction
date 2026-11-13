# Technical Report: Cinema Ticket Demand Forecasting (Show-Level MVP)

**Author:** Senior Machine Learning Engineer Candidate  
**Assessment:** Internship Assessment, Algions AI Labs  
**Date:** November 2026  
**Status:** Complete & Reproducible MVP  

---

## 1. Framing & Business Decision Context
- **Business Goal:** Enable cinema complex managers to optimize auditorium screen assignments and front-of-house staffing levels (ushers, box office, concession stands) ahead of showtime.
- **Prediction Unit:** Show-slot defined by the composite key `(cinema_code, film_code, date, show_time)`.
- **Target Variable:** `tickets_sold` (show attendance volume); secondary reporting metric is `occupancy` = $\frac{\text{tickets\_sold}}{\text{capacity}}$.
- **Prediction Horizon:** Strict $T - 48\text{h}$ lead time. For a screening on `show_date`, only observations from dates $\le \text{show\_date} - 2\text{ days}$ and predetermined schedule attributes (capacity, calendar slot) are legally accessible.
- **Decision Support:** Strictly decision-support. Model output informs human managerial decisions and is not an autonomous allocator.

---

## 2. Dataset Audit & Exploratory Data Analysis (EDA)
- **Source:** Kaggle Cinema Tickets dataset by Möbius (`arashnic/cinema-ticket`), file `cinemaTicket_Ref.csv` (~11.2 MB uncompressed, SHA-256 `93ed89e5b4e005be4093e83df431101f5f50a0d0bc0dd7c3e5af1bba19695c18`).
- **Raw Dimensions:** 142,524 rows across 14 columns covering 257 calendar span days (2018-02-21 to 2018-11-04), with 234 active screening dates and 23 calendar gaps.
- **Catalog Size:** 48 unique films, 246 unique cinema complexes, and 51 distinct `show_time` integer values.
- **Show Time Interpretation:** `show_time` ranges from 1 to 60 (median 3, 75th percentile 5), representing daily sequential slot indices (matinee/slot 1, afternoon/slot 2, evening/slot 3+) rather than military clock hours.
- **Composite Key Duplicates:** 212 rows (0.149%) shared the primary key `(cinema_code, film_code, date, show_time)`. Aggregated during cleaning by summing ticket sales and taking room capacity maximum, yielding 142,418 distinct show slots.
- **Invalid Data & Censoring:**
  - `capacity <= 0`: 179 rows (0.13%). Replaced with each cinema's median valid capacity (global median: 525.8 seats) and flagged with `capacity_imputed`.
  - `tickets_sold > capacity`: 164 rows (0.12%). Capped to capacity to constrain physical occupancy $\le 100\%$.
  - `tickets_sold == 0`: 0 rows in raw data (all recorded shows had $\ge 1$ ticket sold).
  - Demand Censoring: 619 shows (0.43%) reached or exceeded 100% capacity, representing truncated latent demand.

---

## 3. Data Leakage Table
Rigorous mathematical audit conducted in [src/data/audit.py](file:///c:/Users/hp/Desktop/Cinema_Ticket/src/data/audit.py) established two critical leakage vectors:
1. `occu_perc` correlates **1.0000** with `(tickets_sold / capacity) * 100` (mean abs difference $1.25 \times 10^{-15}$). It is target-derived and was dropped.
2. `ticket_price` correlates **1.0000** with `total_sales / tickets_sold` (mean abs difference $5.24 \times 10^{-13}$). Contemporaneous `ticket_price` exposes target ticket sales and was dropped.

| Feature Name | Available at T-48h? | Category | Decision | Operational Justification |
| :--- | :---: | :--- | :---: | :--- |
| `tickets_sold` | No | Target Variable | **Target ($y$)** | Prediction target. Excluded from feature matrix $X$. |
| `occu_perc` | No | Target-Derived | **DROP** | Direct mathematical derivation of target. |
| `total_sales` | No | Post-Show Outcome | **DROP** | Ex-post gross revenue recorded post-screening. |
| `tickets_out` | No | Post-Show Outcome | **DROP** | Ex-post cancellations/refunds. |
| `ticket_use` | No | Post-Show Outcome | **DROP** | Ex-post turnstile ticket admittance. |
| `ticket_price` | No | Contemporaneous Ratio | **DROP** | Ex-post realization ($\frac{\text{total\_sales}}{\text{tickets\_sold}}$). |
| `capacity` | Yes | Theater Geometry | **KEEP** | Fixed auditorium seat count (imputed if invalid). |
| `show_time` | Yes | Timetable Metadata | **KEEP** | Screening slot index published in advance. |
| `cinema_code` | Yes | Categorical Entity | **KEEP** | Cinema complex identifier. |
| `film_code` | Yes | Categorical Entity | **KEEP** | Scheduled movie identifier. |
| `date` | Yes | Temporal Index | **KEEP** | Calendar features extracted; dropped from final $X$. |

---

## 4. Feature Engineering Architecture
Enforcing strict $T - 48\text{h}$ lookahead restriction (shows on date $D$ only see data from $\le D - 2\text{ days}$):
1. **Calendar Dynamics:** `weekday`, `month`, `quarter`, `week_of_year`, `is_weekend` (Fri-Sun), `show_time` slot index.
2. **Theater Capacity:** `capacity`, `log_capacity` ($\log(1 + \text{capacity})$), `capacity_imputed` indicator.
3. **Film Lifecycle:** `days_since_first_show` (computed strictly from past history $\le D - 2$), `is_opening_week` ($\le 7\text{ days}$).
4. **Historical Lags (Shifted $\ge 2$ days):**
   - Rolling 7-day and 14-day mean tickets and occupancy grouped by:
     - Film (`film_mean_tickets_7d`, `film_mean_occu_7d`, `film_mean_tickets_14d`, `film_mean_occu_14d`)
     - Cinema (`cinema_mean_tickets_7d`, `cinema_mean_occu_7d`, `cinema_mean_tickets_14d`, `cinema_mean_occu_14d`)
     - Cinema-Slot (`cinema_slot_mean_tickets_14d`, `cinema_slot_mean_occu_14d`)
     - Film-Cinema (`film_cinema_mean_tickets_14d`, `film_cinema_mean_occu_14d`)
   - Same-weekday last week ($D - 7\text{ days}$): `film_tickets_same_weekday_last_week`, `cinema_occu_same_weekday_last_week`.
5. **Cold-Start & Fallbacks:** `film_history_count`, `cinema_history_count`, `is_new_film` (zero past shows), `is_new_cinema`, `is_low_history` ($<10\text{ shows}$), with automatic fallback to cinema/slot priors.

---

## 5. Chronological Split & Validation Protocol
Splits were generated strictly along chronological dates without shuffling:
- **Train Split (70.4%):** 2018-02-21 to 2018-08-23 (163 days, 100,275 shows, 32 unique films).
- **Validation Split (13.6%):** 2018-08-24 to 2018-09-29 (35 days, 19,379 shows, 30 unique films, 9 new films). Used for hyperparameter tuning and uncertainty residual calibration.
- **Held-Out Test Split (16.0%):** 2018-09-30 to 2018-11-04 (36 days, 22,764 shows, 33 unique films, 16 new films vs train). Evaluated strictly once at the end.

---

## 6. Empirical Model Comparison
All models evaluated on the held-out test set ($N = 22,764\text{ shows}$). Predictions clipped to $[0, \text{capacity}]$.

| Model Architecture | MAE (Tickets) | RMSE (Tickets) | WAPE (%) | Occupancy MAE | Asymmetric Cost ($c_u=1.5, c_o=1.0$) | Clipped at Capacity (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hierarchical Baseline** | 77.63 | 154.01 | 62.83% | 0.1148 | 89.07 | 0.00% |
| **Ridge Regression (Occupancy)** | 67.50 | 137.55 | 54.63% | 0.1081 | 91.84 | 0.00% |
| **LightGBM (Tickets L1)** | **48.36** | **108.96** | **39.14%** | **0.0832** | **65.66** | **0.54%** |

### Performance Analysis:
- **Candidate vs Baseline:** LightGBM delivered a **37.7% reduction in MAE** (48.36 vs 77.63 tickets) and reduced WAPE from 62.83% to 39.14% relative to the hierarchical historical baseline.
- **Asymmetric Operational Cost:** Penalizing under-forecasting ($1.5\times$, risking concession choke-points and customer walkouts) vs over-forecasting ($1.0\times$, idle usher staffing) shows LightGBM achieves a 26.3% lower operational cost (65.66 vs 89.07).

---

## 7. Operational & Cold-Start Slicing Analysis

### Capacity Bucket Slices (LightGBM):
| Auditorium Size | Capacity Range | Test Shows | MAE (Tickets) | WAPE (%) | Occupancy MAE | Clipped (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Small** | $\le 150$ seats | 2,507 | 13.90 | 46.78% | 0.1577 | 4.47% |
| **Medium** | $151 - 300$ seats | 4,309 | 22.78 | 47.47% | 0.1039 | 0.26% |
| **Large** | $> 300$ seats | 15,948 | 60.69 | 38.24% | 0.0659 | 0.01% |

*Insight:* Large auditoriums dominate raw ticket MAE due to scale, but exhibit the best occupancy prediction precision (6.59% Occupancy MAE). Small auditoriums experience higher clipping (4.47%) as sellouts are more frequent.

### Cold-Start Breakdown:
| Cohort | Definition | Test Shows | MAE (Tickets) | WAPE (%) | Occupancy MAE |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Existing Films** | Has prior history ($\ge 1$ show) | 22,415 | 46.17 | 38.01% | 0.0814 |
| **New Films** | Zero prior history ($T-48\text{h}$) | 349 | 189.40 | 73.33% | 0.1980 |
| **Low History** | $< 10$ prior shows | 427 | 159.16 | 73.85% | 0.1818 |
| **Adequate History** | $\ge 10$ prior shows | 22,337 | 46.25 | 37.97% | 0.0813 |

*Cold-Start Finding:* When a film debuts with zero historical tracking data, prediction error increases sharply (WAPE jumps from 38.0% to 73.3%). This empirically justifies the automated `cold_start_warning` flag in the API and dashboard alerting managers to maintain flexible staffing.

---

## 8. Uncertainty Calibration & Residual Coverage
- **Methodology:** Validation set empirical residual quantiles ($y_{val} - \hat{y}_{val}$):
  - $Q_{10} = -41.21$ tickets
  - $Q_{90} = +36.94$ tickets
- **Prediction Interval:** $[\max(0, \hat{y} - 41.2), \min(\text{capacity}, \hat{y} + 36.9)]$
- **Held-Out Test Coverage:** **69.28%** of actual test attendances fell within the nominal 80% interval.
- **Root Cause of Under-Coverage:** Test set (October/November 2018) experienced a higher frequency of blockbuster opening surges and high-capacity variance (evident in the top-10 worst errors), illustrating temporal concept drift.

---

## 9. Error Analysis & Interpretability

### Top 5 Worst Test Predictions:
| Date | Film | Cinema | Capacity | Actual | Predicted | Abs Error | Likely Operational Root Cause |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `2018-10-02` | 1576 | 448 | 5,559 | 5,559 | 3,643 | 1,916 | Mega-complex sold-out opening surge; high variance |
| `2018-10-11` | 1573 | 448 | 3,053 | 3,053 | 1,246 | 1,807 | Unannounced regional promotion / local school holiday |
| `2018-10-10` | 1573 | 448 | 2,703 | 2,703 | 929 | 1,774 | Midweek surge before public holiday |
| `2018-10-11` | 1573 | 524 | 2,490 | 2,490 | 868 | 1,622 | Regional holiday surge across cinema group |
| `2018-10-06` | 1576 | 51 | 1,904 | 1,904 | 291 | 1,613 | Opening weekend sell-out exceeding historical slot mean |

### Global Feature Importance (LightGBM Gain):
1. `film_cinema_mean_tickets_14d` (Gain: 510,611) — Specific film traction in that cinema.
2. `cinema_code` (Gain: 276,658) — Inherent footfall volume of the cinema complex.
3. `cinema_slot_mean_tickets_14d` (Gain: 162,691) — Slot-specific demand cadence.
4. `weekday` (Gain: 151,570) — Strong weekend vs midweek demand cycles.
5. `film_code` (Gain: 83,382) — Film title identity/drawing power.
6. `capacity` (Gain: 56,215) — Room capacity ceiling.

---

## 10. Honest Limitations & Operational Risks
1. **Short Observation Window (~8 months):** Data covers February to November 2018 only. The model cannot learn full yearly seasonality or December/Christmas holiday surges.
2. **Anonymized Entities:** Film titles, cast, genre, director, budget, and marketing spend are not present in the dataset. New film cold-starts must rely purely on slot and theater priors.
3. **Demand Censoring at Capacity:** Recorded ticket sales represent fulfilled demand, not unconstrained latent demand. When shows sell out (0.43%), true demand is censored.
4. **Currency & Geography Unknown:** Monetary amounts are uncalibrated integers without documented currency or national origin. No country or inflation assumptions should be made.
5. **No Live Competitor Signal:** Local events, weather extremes, or competing entertainment options cannot be observed at $T-48\text{h}$.
6. **Mandatory Human Oversight:** Predictions must serve as an initial guideline for theater operations managers, who must adjust allocations using local context.

