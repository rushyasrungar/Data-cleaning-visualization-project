"""
Data-cleaning pipeline for the retail-orders dataset.

Every step is its own small function that takes a DataFrame and returns
(DataFrame, info) so that the notebook can show a before/after for each step
and the log can be written to reports/cleaning_report.md.

Run end-to-end:  python src/clean_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "retail_orders_raw.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "retail_orders_clean.csv"
REPORT_JSON = ROOT / "reports" / "cleaning_summary.json"
REPORT_MD = ROOT / "reports" / "cleaning_report.md"

TEXT_COLS = ["customer_name", "region", "category", "product", "payment_method"]
NUMERIC_COLS = ["customer_age", "quantity", "unit_price", "discount", "delivery_days", "sales", "rating"]
PLACEHOLDERS = {"", "n/a", "na", "none", "null", "nan", "-", "?"}

REGION_MAP = {"n.": "North", "s.": "South", "e.": "East", "w.": "West"}
CATEGORY_MAP = {"electronic": "Electronics", "home and kitchen": "Home & Kitchen", "home & kitchen": "Home & Kitchen"}
PAYMENT_MAP = {"credit card": "Credit Card", "debit card": "Debit Card", "upi": "UPI",
               "net banking": "Net Banking", "cash on delivery": "Cash on Delivery", "cod": "Cash on Delivery"}

# Plausibility rules (domain knowledge): anything outside is a data-entry error, not a real value
VALID_RANGES = {"customer_age": (16, 90), "delivery_days": (1, 30), "discount": (0.0, 0.5),
                "rating": (1, 5), "quantity": (1, 10)}


# --------------------------------------------------------------------------- helpers
def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read everything as text so no information is lost/guessed at load time."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])


def iqr_bounds(s: pd.Series, k: float = 1.5) -> tuple[float, float]:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def count_iqr_outliers(s: pd.Series, k: float = 1.5) -> int:
    s = s.dropna()
    lo, hi = iqr_bounds(s, k)
    return int(((s < lo) | (s > hi)).sum())


def _clean_str(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.strip()
    return s.mask(s.str.lower().isin(PLACEHOLDERS))


def coerce_numeric(s: pd.Series) -> pd.Series:
    """'$1,234.50' -> 1234.5 ; garbage -> NaN."""
    s = s.astype("string").str.replace(r"[$₹,\s]", "", regex=True)
    return pd.to_numeric(s, errors="coerce").astype(float)


def parse_mixed_dates(s: pd.Series) -> pd.Series:
    """Try ISO first, then dd-Mon-YYYY, then dd/mm/YYYY, then mm/dd/YYYY.
    NOTE: dates like 03/04/2024 are genuinely ambiguous; day-first is assumed
    (see README -> Limitations)."""
    s = s.astype("string").str.strip()
    out = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    for fmt in ("%d-%b-%Y", "%d/%m/%Y", "%m/%d/%Y"):
        out = out.fillna(pd.to_datetime(s, format=fmt, errors="coerce"))
    return out


def audit(df: pd.DataFrame) -> dict:
    """Data-quality snapshot of a *raw-as-text* or *clean* frame."""
    d = df.copy()
    for c in TEXT_COLS:
        d[c] = _clean_str(d[c])
    for c in NUMERIC_COLS:
        d[c] = coerce_numeric(d[c]) if d[c].dtype == object or str(d[c].dtype) == "string" or str(d[c].dtype) == "str" else d[c]
    missing = d.isna().sum()
    return {
        "rows": int(len(d)),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_order_ids": int(df["order_id"].duplicated().sum()),
        "missing_by_column": {k: int(v) for k, v in missing.items()},
        "missing_total": int(missing.sum()),
    }


# --------------------------------------------------------------------------- cleaning steps
def standardize_text(df: pd.DataFrame):
    df = df.copy()
    for c in TEXT_COLS:
        df[c] = _clean_str(df[c])

    def apply_map(s: pd.Series, mapping: dict) -> pd.Series:
        mapped = s.str.lower().map(mapping)
        return mapped.fillna(s.str.title()).astype("string")

    before = {c: int(df[c].nunique()) for c in ["region", "category", "payment_method"]}
    df["region"] = apply_map(df["region"], REGION_MAP)
    df["category"] = apply_map(df["category"], CATEGORY_MAP)
    df["payment_method"] = apply_map(df["payment_method"], PAYMENT_MAP)
    df["customer_name"] = df["customer_name"].str.title()
    df["product"] = df["product"].str.title()
    after = {c: int(df[c].nunique()) for c in ["region", "category", "payment_method"]}
    return df, {"unique_labels_before": before, "unique_labels_after": after}


def parse_types(df: pd.DataFrame):
    df = df.copy()
    n_text_sales = int(df["sales"].astype("string").str.contains(r"[$,]", regex=True, na=False).sum())
    bad_dates_raw = df["order_date"]
    df["order_date"] = parse_mixed_dates(df["order_date"])
    unparsed = int(df["order_date"].isna().sum())
    for c in NUMERIC_COLS:
        df[c] = coerce_numeric(df[c])
    df["order_id"] = df["order_id"].astype("string").str.strip()
    return df, {"sales_values_with_currency_symbols": n_text_sales, "unparseable_dates": unparsed}


def remove_duplicates(df: pd.DataFrame):
    n0 = len(df)
    df = df.drop_duplicates()
    exact = n0 - len(df)
    df = df.assign(_nulls=df.isna().sum(axis=1)).sort_values(["order_id", "_nulls"])
    n1 = len(df)
    df = df.drop_duplicates("order_id", keep="first").drop(columns="_nulls")  # keep the most complete record
    same_id = n1 - len(df)
    df = df.sort_values("order_date").reset_index(drop=True)
    return df, {"exact_duplicates_removed": int(exact), "same_order_id_removed": int(same_id), "rows_after": int(len(df))}


def handle_invalid_and_outliers(df: pd.DataFrame):
    """Impossible values -> NaN (imputed later). Extreme-but-possible prices/quantities are
    corrected with robust group medians. Every change is flagged in `outlier_adjusted`."""
    df = df.copy()
    flag = pd.Series(False, index=df.index)
    info: dict[str, int] = {}

    for col, (lo, hi) in VALID_RANGES.items():
        bad = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        info[f"{col}_out_of_range"] = int(bad.sum())
        df.loc[bad, col] = np.nan
        flag |= bad

    # unit_price: compare with the typical price of the same product (or category if product unknown)
    df.loc[df["unit_price"] <= 0, "unit_price"] = np.nan
    prod_med = df.groupby("product")["unit_price"].transform("median")
    cat_med = df.groupby("category")["unit_price"].transform("median")
    ref = prod_med.fillna(cat_med)
    limit = np.where(prod_med.notna(), 3.0, 5.0)
    bad_price = df["unit_price"].notna() & ref.notna() & (df["unit_price"] > limit * ref)
    info["unit_price_extreme"] = int(bad_price.sum())
    df.loc[bad_price, "unit_price"] = ref[bad_price].round(2)
    flag |= bad_price

    df["outlier_adjusted"] = flag
    return df, info


def impute_missing(df: pd.DataFrame):
    df = df.copy()
    info: dict[str, int] = {}
    miss0 = df.isna().sum()

    # 1) infer category from product (a product always belongs to the same category)
    known = df.dropna(subset=["product", "category"])
    prod_to_cat = known.groupby("product")["category"].agg(lambda s: s.mode().iloc[0])
    df["category"] = df["category"].fillna(df["product"].map(prod_to_cat))

    # 2) categorical -> explicit 'Unknown' (inventing a region/payment would bias the analysis)
    for c in ["customer_name", "region", "category", "product", "payment_method"]:
        df[c] = df[c].fillna("Unknown")

    # 3) numeric -> group median (robust to skew), then global median as fallback
    def fill_group_median(col: str, by: str):
        df[col] = df[col].fillna(df.groupby(by)[col].transform("median"))
        df[col] = df[col].fillna(df[col].median())

    fill_group_median("quantity", "product")
    df["quantity"] = df["quantity"].round()                        # whole units, before sales is recomputed
    fill_group_median("unit_price", "product")
    fill_group_median("customer_age", "category")
    fill_group_median("delivery_days", "region")
    df["delivery_days"] = df["delivery_days"].round()
    df["discount"] = df["discount"].fillna(0.0)                    # mode is 0 (no discount)
    fill_group_median("rating", "delivery_days")                   # rating depends on delivery speed
    df["rating"] = (df["rating"] * 2).round() / 2

    # 4) sales must equal quantity * price * (1 - discount); fix text-parse gaps and rows we corrected
    calc = (df["quantity"] * df["unit_price"] * (1 - df["discount"])).round(2)
    mismatch = df["sales"].isna() | ((df["sales"] - calc).abs() > 0.05)
    info["sales_recomputed"] = int(mismatch.sum())
    df["sales"] = calc

    miss1 = df.isna().sum()
    info.update({f"filled_{c}": int(miss0[c] - miss1[c]) for c in df.columns if miss0[c] - miss1[c] > 0})
    return df, info


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["order_date"].dt.year
    df["month"] = df["order_date"].dt.month
    df["quarter"] = "Q" + df["order_date"].dt.quarter.astype(str)
    df["year_month"] = df["order_date"].dt.to_period("M").astype(str)
    df["age_group"] = pd.cut(df["customer_age"], [0, 24, 34, 44, 54, 120],
                             labels=["<25", "25-34", "35-44", "45-54", "55+"]).astype(str)
    df["is_discounted"] = df["discount"] > 0
    df["customer_age"] = df["customer_age"].round().astype(int)
    for c in ["quantity", "delivery_days"]:
        df[c] = df[c].round().astype(int)
    return df


# --------------------------------------------------------------------------- orchestration
def run_cleaning(raw_path: Path = RAW_PATH, out_path: Path = CLEAN_PATH, write: bool = True):
    raw = load_raw(raw_path)
    log: dict = {"before": audit(raw)}

    df, log["standardize_text"] = standardize_text(raw)
    df, log["parse_types"] = parse_types(df)
    num_before = {c: count_iqr_outliers(df[c]) for c in ["quantity", "unit_price", "customer_age", "delivery_days", "sales"]}
    df, log["remove_duplicates"] = remove_duplicates(df)
    df, log["invalid_and_outliers"] = handle_invalid_and_outliers(df)
    df, log["impute_missing"] = impute_missing(df)
    df = add_features(df)

    log["iqr_outliers_before"] = num_before
    log["iqr_outliers_after"] = {c: count_iqr_outliers(df[c]) for c in num_before}
    log["after"] = {"rows": int(len(df)), "missing_total": int(df.isna().sum().sum()),
                    "exact_duplicate_rows": int(df.duplicated().sum()),
                    "duplicate_order_ids": int(df["order_id"].duplicated().sum()),
                    "outlier_adjusted_rows": int(df["outlier_adjusted"].sum())}

    if write:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
        REPORT_JSON.write_text(json.dumps(log, indent=2))
        REPORT_MD.write_text(render_report(log))
    return df, log


def render_report(log: dict) -> str:
    b, a = log["before"], log["after"]
    d, o, m = log["remove_duplicates"], log["invalid_and_outliers"], log["impute_missing"]
    lines = [
        "# Data Cleaning Report", "",
        "| Metric | Before | After |", "|---|---:|---:|",
        f"| Rows | {b['rows']:,} | {a['rows']:,} |",
        f"| Exact duplicate rows | {b['exact_duplicate_rows']:,} | {a['exact_duplicate_rows']} |",
        f"| Duplicate order IDs | {b['duplicate_order_ids']:,} | {a['duplicate_order_ids']} |",
        f"| Missing cells | {b['missing_total']:,} | {a['missing_total']} |", "",
        "## Step by step", "",
        "1. **Standardise text** - trimmed whitespace, unified case, mapped abbreviations "
        f"(unique labels before -> after: {log['standardize_text']['unique_labels_before']} -> {log['standardize_text']['unique_labels_after']}).",
        f"2. **Fix data types** - parsed 4 mixed date formats ({log['parse_types']['unparseable_dates']} unparseable); "
        f"stripped currency symbols from {log['parse_types']['sales_values_with_currency_symbols']:,} `sales` values.",
        f"3. **Remove duplicates** - {d['exact_duplicates_removed']} exact duplicates and {d['same_order_id_removed']} repeated order IDs removed.",
        "4. **Invalid values & outliers** - " + ", ".join(f"{k.replace('_', ' ')}: {v}" for k, v in o.items()) + ".",
        "5. **Missing values** - " + ", ".join(f"{k.replace('filled_', '')}: {v}" for k, v in m.items() if k.startswith('filled_')) + ".",
        f"6. **Sales recomputed** from quantity x price x (1 - discount) for {m['sales_recomputed']} rows.", "",
        "## IQR outlier counts (1.5 x IQR rule)", "", "| Column | Before | After |", "|---|---:|---:|",
    ]
    for c, v in log["iqr_outliers_before"].items():
        lines.append(f"| {c} | {v} | {log['iqr_outliers_after'][c]} |")
    lines += ["", "> Some IQR-flagged rows remain **on purpose**. `sales` (e.g. 5 x smartwatch), `unit_price` "
              "(prices are pooled across very different categories) and `customer_age` (natural tails) "
              "contain genuine values. Errors were caught with domain rules (valid ranges, price vs. product "
              "median), and the plain IQR rule is reported for transparency instead of blindly capping real data.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    df, log = run_cleaning()
    print(render_report(log))
    print(f"Saved clean data -> {CLEAN_PATH}")
