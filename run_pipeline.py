"""One command to reproduce everything:  python run_pipeline.py

1. generate raw data (only if data/raw/retail_orders_raw.csv is missing, or --regenerate)
2. clean it            -> data/processed/retail_orders_clean.csv + reports/cleaning_report.md
3. visualise it        -> reports/figures/*.png, reports/dashboard.html, reports/insights.json
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import clean_data          # noqa: E402
import generate_raw_data   # noqa: E402
import visualize           # noqa: E402


def main():
    if "--regenerate" in sys.argv or not clean_data.RAW_PATH.exists():
        raw = generate_raw_data.make_messy(generate_raw_data.build_clean_truth())
        clean_data.RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(clean_data.RAW_PATH, index=False)
        print(f"[1/3] Raw data generated: {len(raw):,} rows")
    else:
        print("[1/3] Using existing raw data")
    df, log = clean_data.run_cleaning()
    print(f"[2/3] Cleaned: {log['before']['rows']:,} -> {len(df):,} rows, missing cells {log['before']['missing_total']:,} -> {log['after']['missing_total']}")
    visualize.main()
    print("[3/3] Done. Open reports/dashboard.html")


if __name__ == "__main__":
    main()
