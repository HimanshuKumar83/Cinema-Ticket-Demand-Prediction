# Data Card: Cinema Tickets Dataset

## Dataset Summary
- **Dataset Name:** Cinema Tickets
- **Curator / Author:** Möbius (arashnic)
- **Source URL:** [Kaggle - Cinema Ticket](https://www.kaggle.com/datasets/arashnic/cinema-ticket)
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/))
- **File Name:** `cinemaTicket_Ref.csv`
- **File Size:** ~11 MB uncompressed
- **Record Count:** ~142,524 rows
- **Temporal Coverage:** ~8 months (Feb 2018 - Nov 2018)
- **Granularity:** Individual cinema show-level screening sales records

## Licensing & Redistribution Notice
> **IMPORTANT:** In strict adherence to CC BY-NC-SA 4.0 and competitive ML assessment ethics, the raw dataset file (`cinemaTicket_Ref.csv`) is **NOT** tracked or redistributed in this repository. It must be fetched directly from Kaggle via `scripts/download_data.py` using official Kaggle API tokens, or downloaded manually by the user to `data/raw/cinemaTicket_Ref.csv`.

## Business & Problem Framing
- **Decision Context:** Decision-support tool for cinema operations managers allocating screens, theater rooms, and staffing rosters for upcoming shows.
- **Prediction Unit:** Show-slot defined by the composite key `(cinema_code, film_code, date, show_time)`.
- **Target Variable:** `tickets_sold` (the actual attendance for the screening). Secondary metric is `occupancy` = `tickets_sold / capacity`.
- **Prediction Horizon:** 48 hours prior to screening (`show_date - 2`). Any feature computed from tickets or financial totals after `show_date - 2` constitutes data leakage.
- **Ethical / High-Stakes Boundary:** strictly decision-support; human managerial review is required before altering theater schedules or staffing allocations.

## Raw Features & Descriptions
| Column | Type | Description | Usage at T-48h Forecast |
| :--- | :--- | :--- | :--- |
| `film_code` | int/str | Anonymized identifier for the movie | Allowed (Metadata) |
| `cinema_code` | int/str | Anonymized identifier for the cinema complex | Allowed (Metadata) |
| `total_sales` | float | Total monetary revenue for the screening | **LEAKAGE: Dropped** (post-show observation) |
| `tickets_num` | int | Number of tickets sold (`tickets_sold`) | **Target Variable** (y) |
| `tickets_out` | int | Number of refunded/cancelled tickets | **LEAKAGE: Dropped** (post-show observation) |
| `show_time` | int/time | Scheduled screening slot or clock time | Allowed (Known in advance) |
| `occu_perc` | float | Occupancy percentage recorded in source | **LEAKAGE: Dropped** (Directly derived from target) |
| `ticket_price` | float | Price per ticket for the show | **LEAKAGE: Dropped / Lagged only** (Audited for sales dependency) |
| `ticket_use` | int | Number of tickets actually admitted | **LEAKAGE: Dropped** (post-show observation) |
| `capacity` | float | Theater room seat capacity | Allowed (Known in advance, imputed if invalid) |
| `date` | str | Date of the screening (YYYY-MM-DD) | Allowed (Temporal split index) |
| `month` | int | Month index | Allowed (Extracted from date) |
| `quarter` | int | Calendar quarter | Allowed (Extracted from date) |
| `day` | int | Day of month | Allowed (Extracted from date) |

## Data Collection & Privacy
- Film titles, cinema complex names, and geographic locations are fully anonymized numeric codes.
- No Personally Identifiable Information (PII) regarding individual moviegoers exists in this dataset.
- Monetary currency is unspecified in the source documentation; ticket price and revenue should not be presumed in any specific currency.

