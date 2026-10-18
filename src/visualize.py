"""
Visualisations + dashboard for the cleaned retail-orders data.

Run:  python src/visualize.py
Outputs -> reports/figures/*.png , reports/dashboard.html , reports/insights.json
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

import clean_data as cd

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
REPORTS = ROOT / "reports"

BLUE, ORANGE, TEAL, YELLOW, GREEN, PURPLE, GREY = "#2E5EAA", "#E4572E", "#17BEBB", "#F4B942", "#76B041", "#6C4F77", "#8A94A6"
PALETTE = [BLUE, ORANGE, TEAL, YELLOW, GREEN, PURPLE]
sns.set_theme(style="whitegrid", context="notebook", palette=PALETTE)
plt.rcParams.update({"axes.titleweight": "bold", "axes.titlesize": 13, "figure.dpi": 110,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid.axis": "y"})

money = mtick.FuncFormatter(lambda v, _: f"${v/1000:,.0f}k" if abs(v) >= 1000 else f"${v:,.0f}")
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def load_clean() -> pd.DataFrame:
    return pd.read_csv(cd.CLEAN_PATH, parse_dates=["order_date"])


def load_raw_parsed() -> pd.DataFrame:
    """Raw data after type-parsing only (no dedupe / outlier fix) - used for 'before' charts."""
    df, _ = cd.standardize_text(cd.load_raw())
    df, _ = cd.parse_types(df)
    return df


def save(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=130, facecolor="white")
    plt.close(fig)
    return path


# ------------------------------------------------------------------ cleaning-quality charts
def plot_missing_before_after(raw: pd.DataFrame, clean: pd.DataFrame):
    cols = [c for c in raw.columns]
    before = raw[cols].isna().mean() * 100
    after = clean.reindex(columns=cols).isna().mean() * 100
    d = pd.DataFrame({"Before cleaning": before, "After cleaning": after})
    d = d[d["Before cleaning"] > 0].sort_values("Before cleaning")
    fig, ax = plt.subplots(figsize=(8, 5))
    d.plot.barh(ax=ax, color=[ORANGE, TEAL], width=0.75); ax.xaxis.grid(True); ax.yaxis.grid(False)
    ax.set_title("Missing values per column (% of rows)")
    ax.set_xlabel("% missing"); ax.set_ylabel("")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    for p in ax.patches:
        ax.annotate(f"{p.get_width():.1f}%", (p.get_width(), p.get_y() + p.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
    return save(fig, "01_missing_values_before_after.png")


def plot_outliers_before_after(raw: pd.DataFrame, clean: pd.DataFrame):
    cols = [("quantity", "Quantity"), ("unit_price", "Unit price ($)"), ("customer_age", "Customer age"), ("delivery_days", "Delivery days")]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    for j, (c, label) in enumerate(cols):
        sns.boxplot(y=raw[c].dropna(), ax=axes[0, j], color=ORANGE, width=0.45, fliersize=3)
        sns.boxplot(y=clean[c], ax=axes[1, j], color=TEAL, width=0.45, fliersize=3)
        axes[0, j].set_title(label); axes[1, j].set_title("")
        axes[0, j].set_ylabel("Before" if j == 0 else ""); axes[1, j].set_ylabel("After" if j == 0 else "")
    fig.suptitle("Outliers before (top) vs after (bottom) cleaning", fontweight="bold", y=1.0)
    fig.tight_layout()
    return save(fig, "02_outliers_before_after.png")


# ------------------------------------------------------------------ business charts
def plot_monthly_revenue(df: pd.DataFrame):
    m = df.groupby(["year", "month"])["sales"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for (yr, g), col in zip(m.groupby("year"), [GREY, BLUE]):
        ax.plot(g["month"], g["sales"], marker="o", lw=2.5, color=col, label=str(yr))
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTHS)
    ax.yaxis.set_major_formatter(money)
    ax.set_title("Monthly revenue - 2023 vs 2024 (Q4 peak)"); ax.set_xlabel(""); ax.set_ylabel("Revenue")
    ax.legend(title="Year", frameon=False)
    return save(fig, "03_monthly_revenue.png")


def plot_category_region_heatmap(df: pd.DataFrame):
    p = df[df["region"] != "Unknown"].pivot_table(index="category", columns="region", values="sales", aggfunc="sum")
    p = p.loc[p.sum(axis=1).sort_values(ascending=False).index, p.sum().sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.heatmap(p / 1000, annot=True, fmt=".1f", cmap="Blues", cbar_kws={"label": "Revenue ($ thousands)"}, ax=ax, linewidths=.5)
    ax.set_title("Revenue by category and region ($k)"); ax.set_xlabel(""); ax.set_ylabel("")
    return save(fig, "04_category_region_heatmap.png")


def plot_top_products(df: pd.DataFrame):
    t = df[df["product"] != "Unknown"].groupby("product")["sales"].sum().nlargest(10).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(t.index, t.values, color=BLUE)
    ax.xaxis.set_major_formatter(money)
    for y, v in enumerate(t.values):
        ax.text(v, y, f" ${v/1000:,.1f}k", va="center", fontsize=9)
    ax.set_title("Top 10 products by revenue"); ax.set_xlabel("Revenue")
    ax.set_xlim(0, t.max() * 1.15); ax.xaxis.grid(True); ax.yaxis.grid(False)
    return save(fig, "05_top_products.png")


def plot_rating_vs_delivery(df: pd.DataFrame):
    g = df.groupby("delivery_days").agg(rating=("rating", "mean"), n=("rating", "size")).reset_index()
    g = g[g["n"] >= 20]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(g["delivery_days"], g["rating"], marker="o", lw=2.5, color=ORANGE)
    ax.set_ylim(1, 5); ax.set_xlabel("Delivery time (days)"); ax.set_ylabel("Average rating")
    ax.set_title("Slower delivery -> lower customer ratings")
    return save(fig, "06_rating_vs_delivery.png")


def plot_discount_effect(df: pd.DataFrame):
    g = df.assign(bucket=(df["discount"] * 100).round().astype(int).astype(str) + "%") \
          .groupby("bucket").agg(aov=("sales", "mean"), orders=("sales", "size")).reindex(["0%", "5%", "10%", "15%", "20%"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(g.index, g["orders"], color=TEAL)
    ax.set_ylabel("Number of orders"); ax.set_xlabel("Discount given")
    ax2 = ax.twinx(); ax2.plot(g.index, g["aov"], color=ORANGE, marker="o", lw=2.5)
    ax2.set_ylabel("Avg order value ($)", color=ORANGE); ax2.grid(False); ax2.spines["right"].set_visible(True)
    ax.set_title("Discount level: order count (bars) and avg order value (line)")
    return save(fig, "07_discount_effect.png")


def plot_correlation(df: pd.DataFrame):
    cols = ["customer_age", "quantity", "unit_price", "discount", "delivery_days", "rating", "sales"]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, linewidths=.5)
    ax.set_title("Correlation matrix (numeric features)")
    return save(fig, "08_correlation_heatmap.png")


def plot_customer_profile(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(df["customer_age"], bins=20, color=PURPLE, ax=axes[0])
    axes[0].set_title("Customer age distribution"); axes[0].set_xlabel("Age")
    order = ["<25", "25-34", "35-44", "45-54", "55+"]
    g = df.groupby("age_group")["sales"].mean().reindex(order)
    axes[1].bar(g.index, g.values, color=PURPLE)
    axes[1].set_title("Average order value by age group"); axes[1].set_ylabel("$ per order")
    fig.tight_layout()
    return save(fig, "09_customer_profile.png")


# ------------------------------------------------------------------ insights
def compute_insights(df: pd.DataFrame) -> dict:
    total = df["sales"].sum()
    by_year = df.groupby("year")["sales"].sum()
    cat = df.groupby("category")["sales"].sum().sort_values(ascending=False)
    reg = df[df["region"] != "Unknown"].groupby("region")["sales"].sum().sort_values(ascending=False)
    pay = df[df["payment_method"] != "Unknown"]["payment_method"].value_counts(normalize=True)
    prod = df[df["product"] != "Unknown"].groupby("product")["sales"].sum().sort_values(ascending=False)
    q4 = df[df["month"] >= 10]["sales"].sum() / total
    disc = df.groupby("is_discounted")["sales"].mean()
    slow = df[df["delivery_days"] >= 7]["rating"].mean()
    fast = df[df["delivery_days"] <= 3]["rating"].mean()
    return {
        "total_revenue": round(float(total), 2), "orders": int(len(df)), "aov": round(float(df["sales"].mean()), 2),
        "avg_rating": round(float(df["rating"].mean()), 2), "avg_delivery_days": round(float(df["delivery_days"].mean()), 2),
        "pct_discounted": round(float(df["is_discounted"].mean() * 100), 1),
        "revenue_2023": round(float(by_year.get(2023, 0)), 2), "revenue_2024": round(float(by_year.get(2024, 0)), 2),
        "yoy_growth_pct": round(float((by_year.get(2024, 0) / by_year.get(2023, 1) - 1) * 100), 1),
        "top_category": cat.index[0], "top_category_share_pct": round(float(cat.iloc[0] / total * 100), 1),
        "bottom_category": cat.index[-1], "bottom_category_share_pct": round(float(cat.iloc[-1] / total * 100), 1),
        "top_region": reg.index[0], "weakest_region": reg.index[-1],
        "q4_share_pct": round(float(q4 * 100), 1),
        "top_payment": pay.index[0], "top_payment_share_pct": round(float(pay.iloc[0] * 100), 1),
        "top_product": prod.index[0], "top_product_revenue": round(float(prod.iloc[0]), 2),
        "rating_fast_delivery": round(float(fast), 2), "rating_slow_delivery": round(float(slow), 2),
        "aov_discounted": round(float(disc.get(True, 0)), 2), "aov_full_price": round(float(disc.get(False, 0)), 2),
        "corr_delivery_rating": round(float(df["delivery_days"].corr(df["rating"])), 2),
    }


def insight_sentences(i: dict) -> list[str]:
    yoy = i["yoy_growth_pct"]
    trend = (f"Revenue grew {yoy}% year over year" if yoy >= 0 else f"Revenue declined {abs(yoy)}% year over year")
    disc_gap = i["aov_discounted"] - i["aov_full_price"]
    if disc_gap < 0:
        disc = (f"Discounted orders average ${i['aov_discounted']:,.0f} vs ${i['aov_full_price']:,.0f} at full price: "
                f"discounts do not lift basket size, so their profitability should be tested.")
    else:
        disc = (f"Discounted orders average ${i['aov_discounted']:,.0f} vs ${i['aov_full_price']:,.0f} at full price: "
                f"discounts coincide with bigger baskets.")
    return [
        f"{trend} (${i['revenue_2023']:,.0f} in 2023 vs ${i['revenue_2024']:,.0f} in 2024) - growth is flat, so new demand levers are needed.",
        f"Q4 (Oct-Dec) alone brings {i['q4_share_pct']}% of annual revenue (25% would be an even split) - stock and campaigns should peak before October.",
        f"{i['top_category']} is the strongest category ({i['top_category_share_pct']}% of revenue); {i['bottom_category']} is the smallest ({i['bottom_category_share_pct']}%).",
        f"{i['top_region']} leads the regions while {i['weakest_region']} is the weakest - a clear target for regional campaigns.",
        f"Delivery speed drives satisfaction: orders delivered in 3 days or less average a {i['rating_fast_delivery']} rating vs {i['rating_slow_delivery']} for 7+ days (correlation {i['corr_delivery_rating']}).",
        f"{i['top_payment']} is the most used payment method ({i['top_payment_share_pct']}% of orders).",
        disc,
        f"'{i['top_product']}' is the best-selling product by revenue (${i['top_product_revenue']:,.0f}).",
    ]


# ------------------------------------------------------------------ dashboard
def build_dashboard(df: pd.DataFrame, i: dict):
    fig = plt.figure(figsize=(18, 11.5), facecolor="white")
    gs = fig.add_gridspec(3, 6, height_ratios=[0.45, 1.5, 1.4], hspace=0.42, wspace=0.55, top=0.92, bottom=0.06, left=0.05, right=0.98)
    fig.suptitle("Retail Orders Dashboard  |  Jan 2023 - Dec 2024", fontsize=21, fontweight="bold", x=0.02, ha="left", y=0.985)

    kpis = [("Total revenue", f"${i['total_revenue']/1000:,.0f}k", BLUE), ("Orders", f"{i['orders']:,}", TEAL),
            ("Avg order value", f"${i['aov']:,.0f}", PURPLE), ("YoY growth", f"{i['yoy_growth_pct']:+.1f}%", GREEN),
            ("Avg rating", f"{i['avg_rating']:.2f} / 5", YELLOW), ("Avg delivery", f"{i['avg_delivery_days']:.1f} days", ORANGE)]
    for k, (label, val, col) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, k]); ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color=col, alpha=0.12))
        ax.add_patch(plt.Rectangle((0, 0), 0.025, 1, transform=ax.transAxes, color=col))
        ax.text(0.1, 0.62, val, fontsize=19, fontweight="bold", color="#1b1f2a", transform=ax.transAxes, va="center")
        ax.text(0.1, 0.2, label, fontsize=11, color="#4a5160", transform=ax.transAxes, va="center")

    # monthly revenue (2023 vs 2024)
    ax = fig.add_subplot(gs[1, :4])
    m = df.groupby(["year", "month"])["sales"].sum().reset_index()
    for (yr, g), col in zip(m.groupby("year"), [GREY, BLUE]):
        ax.plot(g["month"], g["sales"], marker="o", lw=2.5, color=col, label=str(yr))
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTHS); ax.yaxis.set_major_formatter(money)
    ax.set_title("Monthly revenue: 2023 vs 2024"); ax.legend(frameon=False, loc="upper left")

    # category bars
    ax = fig.add_subplot(gs[1, 4:])
    c = df.groupby("category")["sales"].sum().sort_values()
    ax.barh(c.index, c.values, color=BLUE); ax.xaxis.set_major_formatter(money); ax.set_title("Revenue by category")
    ax.xaxis.set_major_locator(mtick.MaxNLocator(4)); ax.xaxis.grid(True); ax.yaxis.grid(False)

    # region
    ax = fig.add_subplot(gs[2, :2])
    r = df[df["region"] != "Unknown"].groupby("region")["sales"].sum().sort_values(ascending=False)
    ax.bar(r.index, r.values, color=TEAL); ax.yaxis.set_major_formatter(money); ax.set_title("Revenue by region")

    # payment donut
    ax = fig.add_subplot(gs[2, 2:4])
    p = df[df["payment_method"] != "Unknown"]["payment_method"].value_counts()
    ax.pie(p.values, labels=p.index, autopct="%1.0f%%", startangle=90, colors=PALETTE, pctdistance=0.78,
           wedgeprops=dict(width=0.42, edgecolor="white"), textprops={"fontsize": 9})
    ax.set_title("Payment methods")

    # rating vs delivery
    ax = fig.add_subplot(gs[2, 4:])
    g = df.groupby("delivery_days").agg(rating=("rating", "mean"), n=("rating", "size")).reset_index()
    g = g[g["n"] >= 20]
    ax.plot(g["delivery_days"], g["rating"], marker="o", lw=2.5, color=ORANGE)
    ax.set_ylim(1, 5); ax.set_xlabel("Delivery days"); ax.set_ylabel("Avg rating"); ax.set_title("Rating vs delivery time")
    return save(fig, "00_dashboard.png")


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def build_html(i: dict, notes: list[str], figs: dict[str, Path]):
    cards = "".join(f'<div class="card"><div class="v">{v}</div><div class="l">{l}</div></div>' for l, v in [
        ("Total revenue", f"${i['total_revenue']:,.0f}"), ("Orders", f"{i['orders']:,}"), ("Avg order value", f"${i['aov']:,.2f}"),
        ("YoY growth", f"{i['yoy_growth_pct']:+.1f}%"), ("Avg rating", f"{i['avg_rating']:.2f}"), ("Avg delivery", f"{i['avg_delivery_days']:.1f} d")])
    gallery = "".join(f'<figure><img src="data:image/png;base64,{b64(p)}" alt="{t}"><figcaption>{t}</figcaption></figure>'
                      for t, p in figs.items())
    li = "".join(f"<li>{n}</li>" for n in notes)
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Retail Orders - Data Cleaning & Visualization Report</title>
<style>
body{{font-family:system-ui,Segoe UI,Roboto,sans-serif;margin:0;background:#f5f7fb;color:#1b1f2a}}
main{{max-width:1100px;margin:auto;padding:32px 20px}} h1{{margin:0 0 4px}} .sub{{color:#5a6275;margin-bottom:24px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:24px}}
.card{{background:#fff;border-radius:12px;padding:16px;box-shadow:0 1px 3px #0001}} .v{{font-size:24px;font-weight:700;color:#2E5EAA}} .l{{color:#5a6275;font-size:13px}}
section{{background:#fff;border-radius:12px;padding:20px 24px;margin-bottom:24px;box-shadow:0 1px 3px #0001}}
li{{margin:8px 0;line-height:1.5}} img{{max-width:100%;height:auto;border-radius:8px}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px}} figure{{margin:0}} figcaption{{color:#5a6275;font-size:13px;margin-top:6px}}
</style></head><body><main>
<h1>Retail Orders - Data Cleaning &amp; Visualization Report</h1>
<div class="sub">Raw messy CSV &rarr; cleaned dataset &rarr; insights. Jan 2023 - Dec 2024.</div>
<div class="cards">{cards}</div>
<section><h2>Dashboard</h2><img src="data:image/png;base64,{b64(figs.pop('Dashboard'))}" alt="Dashboard"></section>
<section><h2>Key findings</h2><ol>{li}</ol></section>
<section><h2>Charts</h2><div class="gallery">{gallery}</div></section>
</main></body></html>"""
    (REPORTS / "dashboard.html").write_text(html, encoding="utf-8")


def main():
    df, raw = load_clean(), load_raw_parsed()
    insights = compute_insights(df)
    notes = insight_sentences(insights)
    figs = {
        "Dashboard": build_dashboard(df, insights),
        "Missing values: before vs after": plot_missing_before_after(raw, df),
        "Outliers: before vs after": plot_outliers_before_after(raw, df),
        "Monthly revenue": plot_monthly_revenue(df),
        "Category x region heatmap": plot_category_region_heatmap(df),
        "Top products": plot_top_products(df),
        "Rating vs delivery time": plot_rating_vs_delivery(df),
        "Discount effect": plot_discount_effect(df),
        "Correlation matrix": plot_correlation(df),
        "Customer profile": plot_customer_profile(df),
    }
    (REPORTS / "insights.json").write_text(json.dumps({"kpis": insights, "findings": notes}, indent=2))
    build_html(insights, notes, dict(figs))
    print(f"Saved {len(figs)} figures + dashboard.html")
    for n in notes:
        print(" -", n)


if __name__ == "__main__":
    main()
