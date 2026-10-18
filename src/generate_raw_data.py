"""
Generate a realistic, deliberately MESSY retail-orders dataset.

Why synthetic? It lets the project ship with its own raw data, and lets us
inject every kind of real-world problem on purpose (missing values, duplicates,
outliers, inconsistent text, mixed date formats, currency symbols...), so each
cleaning step in clean_data.py has something real to fix.

Run:  python src/generate_raw_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 5000
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "retail_orders_raw.csv"

rng = np.random.default_rng(SEED)

CATALOG = {
    "Electronics": {"Headphones": 60, "Smartwatch": 150, "Bluetooth Speaker": 45, "Power Bank": 25, "Webcam": 40},
    "Clothing": {"T-Shirt": 15, "Jeans": 45, "Hoodie": 38, "Sneakers": 70, "Jacket": 95},
    "Home & Kitchen": {"Air Fryer": 85, "Blender": 50, "Cookware Set": 110, "Coffee Maker": 65, "Vacuum Cleaner": 140},
    "Books": {"Novel": 12, "Textbook": 55, "Comic": 9, "Cookbook": 22, "Biography": 18},
    "Sports": {"Yoga Mat": 20, "Dumbbell Set": 60, "Cycling Helmet": 42, "Football": 25, "Tennis Racket": 80},
}
CAT_WEIGHTS = [0.28, 0.24, 0.20, 0.12, 0.16]
REGIONS = ["North", "South", "East", "West", "Central"]
PAYMENTS = ["Credit Card", "Debit Card", "UPI", "Net Banking", "Cash on Delivery"]
FIRST = ["Aarav", "Priya", "Rohan", "Sneha", "Karthik", "Ananya", "Vikram", "Divya", "Arjun", "Meera",
         "Rahul", "Kavya", "Siddharth", "Isha", "Nikhil", "Pooja", "Aditya", "Neha", "Suresh", "Lakshmi"]
LAST = ["Sharma", "Iyer", "Reddy", "Nair", "Patel", "Singh", "Kumar", "Menon", "Gupta", "Das"]


def build_clean_truth() -> pd.DataFrame:
    dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 730, N_ROWS), unit="D")
    # seasonality: more orders in Oct-Dec
    keep = rng.random(N_ROWS) < np.where(dates.month >= 10, 1.0, 0.75)
    dates = dates[keep]
    n = len(dates)

    cats = rng.choice(list(CATALOG), size=n, p=CAT_WEIGHTS)
    products = np.array([rng.choice(list(CATALOG[c])) for c in cats])
    base_price = np.array([CATALOG[c][p] for c, p in zip(cats, products)], dtype=float)
    unit_price = np.round(base_price * rng.normal(1.0, 0.06, n), 2)

    df = pd.DataFrame({
        "order_id": [f"ORD-{100000 + i}" for i in range(n)],
        "order_date": dates,
        "customer_name": [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(n)],
        "customer_age": np.clip(rng.normal(34, 11, n), 18, 75).round().astype(int),
        "region": rng.choice(REGIONS, size=n, p=[0.22, 0.25, 0.20, 0.23, 0.10]),
        "category": cats,
        "product": products,
        "quantity": rng.choice([1, 2, 3, 4, 5], size=n, p=[0.45, 0.28, 0.14, 0.08, 0.05]),
        "unit_price": unit_price,
        "discount": rng.choice([0.0, 0.05, 0.10, 0.15, 0.20], size=n, p=[0.45, 0.2, 0.18, 0.1, 0.07]),
        "payment_method": rng.choice(PAYMENTS, size=n, p=[0.28, 0.22, 0.30, 0.10, 0.10]),
        "delivery_days": np.clip(rng.normal(4.5, 1.8, n), 1, 12).round().astype(int),
    })
    df["sales"] = (df["quantity"] * df["unit_price"] * (1 - df["discount"])).round(2)
    # rating loosely tied to delivery speed
    rating = 4.6 - 0.18 * (df["delivery_days"] - 3) + rng.normal(0, 0.6, n)
    df["rating"] = np.clip(np.round(rating * 2) / 2, 1, 5)
    return df


def make_messy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    # 1. Mixed date formats
    fmts = ["%Y-%m-%d", "%d/%m/%Y", "%d-%b-%Y", "%m/%d/%Y"]
    pick = rng.choice(len(fmts), size=n, p=[0.55, 0.20, 0.15, 0.10])
    df["order_date"] = [d.strftime(fmts[k]) for d, k in zip(df["order_date"], pick)]

    # 2. Inconsistent text (case / whitespace / abbreviations)
    def mess_text(s, abbrev=None):
        r = rng.random()
        if r < 0.10:
            return s.upper()
        if r < 0.20:
            return s.lower()
        if r < 0.28:
            return f"  {s} "
        if abbrev and r < 0.33:
            return abbrev.get(s, s)
        return s

    df["region"] = [mess_text(s, {"North": "N.", "South": "S.", "East": "E.", "West": "W."}) for s in df["region"]]
    df["category"] = [mess_text(s, {"Home & Kitchen": "Home and Kitchen", "Electronics": "Electronic"}) for s in df["category"]]
    df["payment_method"] = [mess_text(s, {"UPI": "Upi", "Cash on Delivery": "COD"}) for s in df["payment_method"]]
    df["customer_name"] = [mess_text(s) for s in df["customer_name"]]

    # 3. Currency symbols / thousands separators stored as text in 'sales'
    sales = df["sales"].astype(object)
    idx = rng.choice(n, size=int(0.12 * n), replace=False)
    for i in idx:
        v = sales.iloc[i]
        sales.iloc[i] = f"${v:,.2f}" if rng.random() < 0.7 else f"{v:,.2f}"
    df["sales"] = sales

    # 4. Outliers & impossible values
    q_idx = rng.choice(n, size=25, replace=False)
    df.loc[df.index[q_idx], "quantity"] = rng.choice([40, 60, 99, 150], size=25)
    p_idx = rng.choice(n, size=20, replace=False)
    df.loc[df.index[p_idx], "unit_price"] = (df["unit_price"].iloc[p_idx] * rng.choice([12, 20, 35], size=20)).round(2)
    a_idx = rng.choice(n, size=15, replace=False)
    df.loc[df.index[a_idx], "customer_age"] = rng.choice([0, 2, 150, 199, 250, -5], size=15)
    d_idx = rng.choice(n, size=15, replace=False)
    df.loc[df.index[d_idx], "delivery_days"] = rng.choice([-3, 45, 90, 120], size=15)

    # 5. Missing values (different rates per column, like real data)
    for col, rate in {"customer_age": 0.07, "region": 0.04, "rating": 0.10, "payment_method": 0.03,
                      "discount": 0.05, "delivery_days": 0.04, "customer_name": 0.02}.items():
        m = rng.random(n) < rate
        df[col] = df[col].astype(object)
        df.loc[m, col] = np.nan
    blank = rng.choice(n, size=40, replace=False)
    df.loc[df.index[blank], "product"] = ""          # blanks that are *not* NaN
    df.loc[df.index[rng.choice(n, size=25, replace=False)], "category"] = "N/A"

    # 6. Duplicates: exact copies + the same order_id re-entered
    exact = df.sample(int(0.03 * n), random_state=SEED)
    dup_id = df.sample(int(0.01 * n), random_state=SEED + 1).copy()
    dup_id["customer_name"] = dup_id["customer_name"].astype(object)
    df = pd.concat([df, exact, dup_id], ignore_index=True)
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


if __name__ == "__main__":
    truth = build_clean_truth()
    raw = make_messy(truth)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(OUT, index=False)
    print(f"Wrote {len(raw):,} raw rows -> {OUT}")
