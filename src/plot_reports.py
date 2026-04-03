"""
plot_reports.py — Comprehensive visualisation suite.

Generates 8 publication-quality plots saved to outputs/plots/.
Run after the full pipeline (forecast → pvi → recommend → evaluate).

Plots generated
---------------
1. forecast_vs_actual_top5.png  — Forecast vs actual sales for top-5 PVI items
2. mae_rmse_comparison.png      — Grouped bar: MAE & RMSE, Prophet vs ARIMA
3. mape_wape_comparison.png     — Grouped bar: MAPE & WAPE, Prophet vs ARIMA
4. r2_comparison.png            — Box plot: R² distribution per model
5. residual_distribution.png    — Histogram of forecast errors per model
6. actual_vs_predicted.png      — Scatter: actual vs predicted sales
7. pvi_distribution.png         — PVI histogram coloured by viability category
8. pvi_by_category.png          — Avg PVI per product category with breakdown
9. decision_breakdown.png       — Stacked bar: decisions per store

Usage
-----
    python src/plot_reports.py
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")
matplotlib.use("Agg")   # non-interactive backend — safe for servers

# ── Paths ──────────────────────────────────────────────────────────────────
PROCESSED_PATH       = "data/processed/processed_m5.csv"
PROPHET_FORECAST_DIR = "data/forecast/prophet"
ARIMA_FORECAST_DIR   = "data/forecast/arima"
PVI_PATH             = "data/pvi_scores.csv"
RECS_PATH            = "data/recommendations.csv"
EVAL_PATH            = "data/eval_metrics.csv"
EVAL_SUMMARY_PATH    = "data/eval_summary.csv"
OUT_DIR              = "outputs/plots"

# ── Design tokens ──────────────────────────────────────────────────────────
PROPHET_COLOR  = "#3b82f6"   # blue
ARIMA_COLOR    = "#7c3aed"   # purple
HIGH_COLOR     = "#15803d"   # green
MEDIUM_COLOR   = "#b45309"   # amber
LOW_COLOR      = "#b91c1c"   # red
INCREASE_COLOR = "#1d4ed8"
HOLD_COLOR     = "#6d28d9"
DECREASE_COLOR = "#b91c1c"
GRID_COLOR     = "#f5f5f4"
TEXT_COLOR     = "#1c1917"
MUTED_COLOR    = "#78716c"

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#e7e5e4",
    "axes.labelcolor":   TEXT_COLOR,
    "xtick.color":       MUTED_COLOR,
    "ytick.color":       MUTED_COLOR,
    "text.color":        TEXT_COLOR,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "grid.color":        GRID_COLOR,
    "grid.linewidth":    1,
})


def _save(fig, name: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {path}")


def _parse_filename(fname: str):
    base      = os.path.basename(fname).replace(".csv", "")
    rest      = base[len("forecast_"):]
    parts     = rest.split("_")
    store_id  = f"{parts[0]}_{parts[1]}"
    item_id   = "_".join(parts[2:]).replace("-", "_")
    return store_id, item_id


# ---------------------------------------------------------------------------
# Plot 1 — Forecast vs Actual (top 5 PVI items)
# ---------------------------------------------------------------------------

def plot_forecast_vs_actual():
    if not os.path.exists(PVI_PATH) or not os.path.exists(PROCESSED_PATH):
        print("  Skipping forecast_vs_actual — PVI or processed data missing")
        return

    pvi_df    = pd.read_csv(PVI_PATH)
    processed = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
    top5      = pvi_df.nlargest(5, "PVI")[["store_id", "item_id", "PVI", "viability"]]

    fig, axes = plt.subplots(5, 1, figsize=(12, 18))
    fig.suptitle("Forecast vs Actual Sales — Top 5 Items by PVI", fontsize=15, fontweight="bold", y=0.98)

    for ax, (_, row) in zip(axes, top5.iterrows()):
        store, item = row["store_id"], row["item_id"]
        safe_item   = item.replace("_", "-")

        hist = processed[(processed["store_id"] == store) & (processed["item_id"] == item)].sort_values("date")

        prophet_path = os.path.join(PROPHET_FORECAST_DIR, f"forecast_{store}_{safe_item}.csv")
        arima_path   = os.path.join(ARIMA_FORECAST_DIR,   f"forecast_{store}_{safe_item}.csv")

        ax.plot(hist["date"], hist["monthly_sales"], color=TEXT_COLOR,
                linewidth=2, label="Actual", zorder=3)

        if os.path.exists(prophet_path):
            fc = pd.read_csv(prophet_path, parse_dates=["ds"])
            ax.plot(fc["ds"], fc["yhat"], color=PROPHET_COLOR, linewidth=2,
                    linestyle="--", label="Prophet", zorder=2)
            if "yhat_lower" in fc.columns:
                ax.fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"],
                                alpha=0.15, color=PROPHET_COLOR)

        if os.path.exists(arima_path):
            fc = pd.read_csv(arima_path, parse_dates=["ds"])
            ax.plot(fc["ds"], fc["yhat"], color=ARIMA_COLOR, linewidth=2,
                    linestyle=":", label="ARIMA", zorder=2)
            if "yhat_lower" in fc.columns:
                ax.fill_between(fc["ds"], fc["yhat_lower"], fc["yhat_upper"],
                                alpha=0.10, color=ARIMA_COLOR)

        vcolor = HIGH_COLOR if row["viability"] == "High" else (MEDIUM_COLOR if row["viability"] == "Medium" else LOW_COLOR)
        ax.set_title(f"{item}  |  {store}  |  PVI {row['PVI']:.1f}  ({row['viability']})",
                     color=vcolor, fontsize=11)
        ax.set_ylabel("Monthly sales")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.8)

    axes[-1].set_xlabel("Date")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, "forecast_vs_actual_top5.png")


# ---------------------------------------------------------------------------
# Plot 2 — MAE & RMSE comparison
# ---------------------------------------------------------------------------

def plot_mae_rmse_comparison():
    if not os.path.exists(EVAL_SUMMARY_PATH):
        print("  Skipping mae_rmse — eval_summary missing"); return

    df = pd.read_csv(EVAL_SUMMARY_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Forecast Error — Prophet vs ARIMA", fontsize=14, fontweight="bold")

    for ax, metric in zip(axes, ["MAE", "RMSE"]):
        sub     = df[df["metric"] == metric]
        models  = sub["model"].tolist()
        means   = sub["mean"].tolist()
        stds    = sub["std"].tolist()
        colors  = [PROPHET_COLOR if m == "prophet" else ARIMA_COLOR for m in models]
        bars    = ax.bar(models, means, color=colors, width=0.5, zorder=2, edgecolor="white")
        ax.errorbar(models, means, yerr=stds, fmt="none", color="#44403c",
                    capsize=5, linewidth=1.5, zorder=3)

        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(stds) * 0.05,
                    f"{val:,.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_title(f"{metric}  (lower is better)", pad=10)
        ax.set_ylabel("Units of sales")
        ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
        ax.set_ylim(0, max(means) * 1.3)

    fig.tight_layout()
    _save(fig, "mae_rmse_comparison.png")


# ---------------------------------------------------------------------------
# Plot 3 — MAPE & WAPE comparison
# ---------------------------------------------------------------------------

def plot_mape_wape_comparison():
    if not os.path.exists(EVAL_SUMMARY_PATH):
        print("  Skipping mape_wape — eval_summary missing"); return

    df = pd.read_csv(EVAL_SUMMARY_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("Percentage Error — Prophet vs ARIMA", fontsize=14, fontweight="bold")

    for ax, metric, note in zip(axes, ["MAPE", "WAPE"],
                                 ["skips zero-actual months", "weighted by sales volume"]):
        sub   = df[df["metric"] == metric]
        if sub.empty:
            ax.text(0.5, 0.5, f"{metric} not available", ha="center", va="center",
                    transform=ax.transAxes, color=MUTED_COLOR)
            continue

        models = sub["model"].tolist()
        means  = sub["mean"].tolist()
        colors = [PROPHET_COLOR if m == "prophet" else ARIMA_COLOR for m in models]
        bars   = ax.bar(models, means, color=colors, width=0.5, zorder=2, edgecolor="white")

        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_title(f"{metric}  —  {note}", pad=10)
        ax.set_ylabel("Percentage error (%)")
        ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
        ax.set_ylim(0, max(means) * 1.35)

    fig.tight_layout()
    _save(fig, "mape_wape_comparison.png")


# ---------------------------------------------------------------------------
# Plot 4 — R² distribution
# ---------------------------------------------------------------------------

def plot_r2_distribution():
    if not os.path.exists(EVAL_PATH):
        print("  Skipping r2_distribution — eval_metrics missing"); return

    df = pd.read_csv(EVAL_PATH)
    if "R2" not in df.columns:
        print("  Skipping r2_distribution — R2 column missing"); return

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("R² Distribution — How Well Each Model Explains Variance",
                 fontsize=14, fontweight="bold")

    data   = [df[df["model"] == m]["R2"].dropna().values for m in ["prophet", "arima"]]
    labels = ["Prophet", "ARIMA"]
    colors = [PROPHET_COLOR, ARIMA_COLOR]

    bp = ax.boxplot(data, patch_artist=True, widths=0.4,
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5),
                    flierprops=dict(marker="o", markersize=4, alpha=0.4))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.axhline(0, color="#e24b4a", linestyle="--", linewidth=1.2,
               label="R²=0 (no better than mean)")
    ax.axhline(1, color=HIGH_COLOR, linestyle="--", linewidth=1.2,
               label="R²=1 (perfect forecast)")

    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels)
    ax.set_ylabel("R² score")

    # FIX 1: Expand ylim to show actual data range, not cut it off
    all_vals = [v for d in data for v in d]
    y_min = max(np.percentile(all_vals, 1), -15)   # clip extreme outliers for display
    ax.set_ylim(y_min - 0.5, 1.3)

    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, framealpha=0.8)

    # FIX 2: Clamp text position inside axes bounds so it doesn't blow up the canvas
    y_lo, y_hi = ax.get_ylim()
    for i, (d, color) in enumerate(zip(data, colors), 1):
        if len(d):
            med = np.median(d)
            # Place label inside axes — if median is below view, pin to bottom
            text_y = np.clip(med, y_lo + 0.3, y_hi - 0.1)
            label  = f"  med={med:.2f}"
            if med < y_lo + 0.3:
                label = f"  med={med:.2f} (below view)"
            ax.text(i, text_y, label,
                    va="center", fontsize=9, color=color, fontweight="bold")

    fig.tight_layout()
    _save(fig, "r2_distribution.png")


# ---------------------------------------------------------------------------
# Plot 5 — Residual distribution
# ---------------------------------------------------------------------------

def plot_residual_distribution():
    if not os.path.exists(EVAL_PATH):
        print("  Skipping residuals — eval_metrics missing"); return

    df = pd.read_csv(EVAL_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    fig.suptitle("Residual Distribution (Forecast Error = Predicted − Actual)",
                 fontsize=14, fontweight="bold")

    for ax, (model, color) in zip(axes, [("prophet", PROPHET_COLOR), ("arima", ARIMA_COLOR)]):
        sub = df[df["model"] == model]["bias"].dropna()
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        ax.hist(sub, bins=25, color=color, alpha=0.8, edgecolor="white", zorder=2)
        ax.axvline(0,          color=TEXT_COLOR,   linestyle="--", linewidth=1.5, label="Zero bias")
        ax.axvline(sub.mean(), color="#e24b4a",    linestyle="-",  linewidth=2,   label=f"Mean = {sub.mean():.1f}")

        ax.set_title(f"{model.capitalize()}  —  bias distribution")
        ax.set_xlabel("Forecast error (predicted − actual units)")
        ax.set_ylabel("Number of items")
        ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
        ax.legend(fontsize=9)

        note = "Over-predicts" if sub.mean() > 0 else "Under-predicts"
        ax.text(0.97, 0.95, note, transform=ax.transAxes,
                ha="right", va="top", fontsize=10,
                color="#e24b4a" if sub.mean() > 0 else ARIMA_COLOR,
                fontweight="bold")

    fig.tight_layout()
    _save(fig, "residual_distribution.png")


# ---------------------------------------------------------------------------
# Plot 6 — Actual vs Predicted scatter
# ---------------------------------------------------------------------------

def plot_actual_vs_predicted():
    if not os.path.exists(EVAL_PATH):
        print("  Skipping actual_vs_predicted — eval_metrics missing"); return

    df = pd.read_csv(EVAL_PATH)
    if "y_true_mean" not in df.columns:
        print("  Skipping actual_vs_predicted — y_true_mean column missing"); return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Actual vs Predicted Sales — Perfect = Points on Diagonal",
                 fontsize=14, fontweight="bold")

    for ax, (model, color) in zip(axes, [("prophet", PROPHET_COLOR), ("arima", ARIMA_COLOR)]):
        sub = df[df["model"] == model].dropna(subset=["y_true_mean", "y_pred_mean"])
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        x, y = sub["y_true_mean"].values, sub["y_pred_mean"].values
        ax.scatter(x, y, alpha=0.35, color=color, s=18, zorder=2, edgecolors="none")

        lo, hi = min(x.min(), y.min()) * 0.9, max(x.max(), y.max()) * 1.1
        ax.plot([lo, hi], [lo, hi], color=TEXT_COLOR, linewidth=1.5,
                linestyle="--", label="Perfect forecast")

        # Regression line
        if len(x) > 2:
            m, b = np.polyfit(x, y, 1)
            xs   = np.linspace(lo, hi, 100)
            ax.plot(xs, m * xs + b, color=color, linewidth=2,
                    label=f"Fitted (slope={m:.2f})")

        ax.set_title(f"{model.capitalize()}")
        ax.set_xlabel("Actual monthly sales")
        ax.set_ylabel("Predicted monthly sales")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(linestyle="--", alpha=0.4, zorder=0)
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    fig.tight_layout()
    _save(fig, "actual_vs_predicted.png")


# ---------------------------------------------------------------------------
# Plot 7 — PVI distribution
# ---------------------------------------------------------------------------

def plot_pvi_distribution():
    if not os.path.exists(PVI_PATH):
        print("  Skipping pvi_distribution — pvi_scores missing"); return

    pvi = pd.read_csv(PVI_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Product Viability Index (PVI) Distribution", fontsize=14, fontweight="bold")

    # Left: overall histogram coloured by viability zone
    ax = axes[0]
    bins = np.linspace(0, 100, 26)
    for viab, color in [("High", HIGH_COLOR), ("Medium", MEDIUM_COLOR), ("Low", LOW_COLOR)]:
        sub = pvi[pvi["viability"] == viab]["PVI"]
        ax.hist(sub, bins=bins, color=color, alpha=0.8, edgecolor="white", label=viab, zorder=2)

    ax.axvline(33, color=MUTED_COLOR, linestyle="--", linewidth=1, label="Low/Medium boundary")
    ax.axvline(67, color=MUTED_COLOR, linestyle=":",  linewidth=1, label="Medium/High boundary")
    ax.set_xlabel("PVI score (0–100)")
    ax.set_ylabel("Number of items")
    ax.set_title("All items by viability zone")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    counts = pvi["viability"].value_counts()
    total  = len(pvi)
    for viab, color, y_pos in [("High", HIGH_COLOR, 0.90), ("Medium", MEDIUM_COLOR, 0.82), ("Low", LOW_COLOR, 0.74)]:
        n = counts.get(viab, 0)
        ax.text(0.97, y_pos, f"{viab}: {n} ({100*n/total:.0f}%)",
                transform=ax.transAxes, ha="right", fontsize=10,
                color=color, fontweight="bold")

    # Right: PVI by category
    ax2 = axes[1]
    if "cat_id" in pvi.columns:
        cat_means = pvi.groupby(["cat_id", "viability"]).size().unstack(fill_value=0)
        cat_order = pvi.groupby("cat_id")["PVI"].mean().sort_values(ascending=False).index
        cat_means = cat_means.reindex(cat_order)

        bottom = np.zeros(len(cat_means))
        for viab, color in [("High", HIGH_COLOR), ("Medium", MEDIUM_COLOR), ("Low", LOW_COLOR)]:
            if viab in cat_means.columns:
                vals = cat_means[viab].values
                ax2.bar(cat_means.index, vals, bottom=bottom,
                        color=color, label=viab, edgecolor="white")
                bottom += vals

        avg_pvi = pvi.groupby("cat_id")["PVI"].mean().reindex(cat_order)
        for i, (cat, val) in enumerate(avg_pvi.items()):
            ax2.text(i, bottom[i] + 1, f"avg\n{val:.0f}", ha="center",
                     va="bottom", fontsize=9, color=TEXT_COLOR)

    ax2.set_title("Items per category by viability")
    ax2.set_xlabel("Category")
    ax2.set_ylabel("Number of items")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    fig.tight_layout()
    _save(fig, "pvi_distribution.png")


# ---------------------------------------------------------------------------
# Plot 8 — PVI sub-score heatmap by category
# ---------------------------------------------------------------------------

def plot_pvi_subscores_by_category():
    if not os.path.exists(PVI_PATH):
        print("  Skipping subscores heatmap — pvi_scores missing"); return

    pvi = pd.read_csv(PVI_PATH)
    if "cat_id" not in pvi.columns:
        print("  Skipping subscores heatmap — cat_id missing"); return

    sub_cols = ["demand_norm", "growth_norm", "stability_norm", "price_norm"]
    missing  = [c for c in sub_cols if c not in pvi.columns]
    if missing:
        print(f"  Skipping subscores heatmap — missing columns: {missing}"); return

    pivot = pvi.groupby("cat_id")[sub_cols].mean() * 100
    pivot.columns = ["Demand (40%)", "Growth (25%)", "Stability (20%)", "Price (15%)"]

    fig, ax = plt.subplots(figsize=(9, max(4, len(pivot) * 1.1)))
    fig.suptitle("Average PVI Sub-scores by Category", fontsize=14, fontweight="bold")

    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=15, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if val < 35 or val > 70 else TEXT_COLOR)

    plt.colorbar(im, ax=ax, label="Score (0–100%)", shrink=0.8)
    fig.tight_layout()
    _save(fig, "pvi_subscores_heatmap.png")


# ---------------------------------------------------------------------------
# Plot 9 — Decision breakdown per store
# ---------------------------------------------------------------------------

def plot_decision_breakdown():
    if not os.path.exists(RECS_PATH):
        print("  Skipping decision_breakdown — recommendations missing"); return

    recs = pd.read_csv(RECS_PATH)
    if "store_id" not in recs.columns:
        print("  Skipping decision_breakdown — store_id column missing"); return

    pivot = recs.groupby(["store_id", "Decision"]).size().unstack(fill_value=0)
    for col in ["Increase", "Hold", "Decrease"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["Increase", "Hold", "Decrease"]]
    pivot = pivot.sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Stock Recommendation Breakdown", fontsize=14, fontweight="bold")

    # Stacked bar per store
    ax = axes[0]
    bottom = np.zeros(len(pivot))
    for dec, color in [("Increase", INCREASE_COLOR), ("Hold", HOLD_COLOR), ("Decrease", DECREASE_COLOR)]:
        vals = pivot[dec].values
        bars = ax.bar(pivot.index, vals, bottom=bottom, color=color,
                      label=dec, edgecolor="white")
        for bar, val, bot in zip(bars, vals, bottom):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bot + val / 2,
                        str(val), ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
        bottom += vals

    ax.set_title("Decisions per store")
    ax.set_xlabel("Store ID")
    ax.set_ylabel("Number of items")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")

    # Overall donut
    ax2 = axes[1]
    totals = pivot.sum()
    colors = [INCREASE_COLOR, HOLD_COLOR, DECREASE_COLOR]
    wedges, texts, autotexts = ax2.pie(
        totals.values, labels=totals.index, autopct="%1.0f%%",
        colors=colors, startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2),
        textprops=dict(fontsize=11)
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_color("white")
        at.set_fontweight("bold")

    # Draw hole for donut effect
    centre_circle = plt.Circle((0, 0), 0.60, fc="white")
    ax2.add_patch(centre_circle)
    ax2.set_title("Overall decision split")
    ax2.text(0, 0, f"{len(recs)}\nitems", ha="center", va="center",
             fontsize=12, fontweight="bold", color=TEXT_COLOR)

    fig.tight_layout()
    _save(fig, "decision_breakdown.png")


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------

def generate_all_plots():
    print(f"\n{'='*55}")
    print("Generating all visualisation plots…")
    print(f"{'='*55}")

    plot_forecast_vs_actual()
    plot_mae_rmse_comparison()
    plot_mape_wape_comparison()
    plot_r2_distribution()
    plot_residual_distribution()
    plot_actual_vs_predicted()
    plot_pvi_distribution()
    plot_pvi_subscores_by_category()
    plot_decision_breakdown()

    print(f"\nAll plots saved to  {OUT_DIR}/")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    generate_all_plots()