import os
import numpy as np
import pandas as pd

PVI_PATH    = "data/pvi_scores.csv"
OUTPUT_PATH = "data/recommendations.csv"

PVI_HIGH   = 67
PVI_MEDIUM = 33

GROWTH_POSITIVE_THRESHOLD = 0.55
GROWTH_NEGATIVE_THRESHOLD = 0.40

STABILITY_LOW_THRESHOLD = 0.40

ANOMALY_SERIOUS_THRESHOLD = 0.15

AGREEMENT_LOW_THRESHOLD  = 0.55
AGREEMENT_HIGH_THRESHOLD = 0.80


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def compute_confidence(row: pd.Series) -> str:

    score = 1.0

    agreement = row.get("model_agreement")
    if pd.notna(agreement):
        if agreement < AGREEMENT_LOW_THRESHOLD:
            score -= 0.4
        elif agreement < AGREEMENT_HIGH_THRESHOLD:
            score -= 0.15

    if row.get("has_anomaly", False):
        score -= 0.2

    pvi = row["PVI"]
    dist_to_boundary = min(abs(pvi - PVI_HIGH), abs(pvi - PVI_MEDIUM))
    if dist_to_boundary < 5:
        score -= 0.2

    if score >= 0.75:
        return "High"
    if score >= 0.45:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Explanation builder
# ---------------------------------------------------------------------------

def build_explanation(
    decision: str,
    row: pd.Series,
    growth_label: str,
    risk_label: str,
    anomaly_override: bool,
) -> str:

    parts = []

    pvi        = row["PVI"]
    viability  = row["viability"]
    demand_n   = row.get("demand_norm",    np.nan)
    growth_n   = row.get("growth_norm",    np.nan)
    stability_n= row.get("stability_norm", np.nan)
    price_n    = row.get("price_norm",     np.nan)

    parts.append(
        f"PVI={pvi:.1f}/100 ({viability} viability): "
        f"demand={demand_n:.2f}, growth={growth_n:.2f}, "
        f"stability={stability_n:.2f}, price={price_n:.2f}"
    )

    if row.get("has_anomaly", False):
        pct = row.get("anomaly_pct", 0) * 100
        parts.append(
            f"⚠ Anomaly detected in {pct:.0f}% of historical months"
            + (" — override applied" if anomaly_override else "")
        )

    if decision == "Increase":
        parts.append(
            f"Recommend stocking up: {growth_label} demand trend, "
            f"{risk_label} supply risk, and high viability support expansion."
        )
    elif decision == "Decrease":
        if viability == "Low":
            parts.append(
                f"Low viability with {growth_label} trend — "
                f"reducing stock minimises overstock waste."
            )
        else:
            parts.append(
                f"High supply risk ({risk_label}) despite moderate viability — "
                f"hold back until demand stabilises."
            )
    else:
        parts.append(
            f"Maintain current stock levels: {growth_label} trend, "
            f"{risk_label} risk. Monitor PVI trajectory."
        )

    agreement = row.get("model_agreement")
    if pd.notna(agreement):
        if agreement < AGREEMENT_LOW_THRESHOLD:
            parts.append(
                f"⚠ Prophet and ARIMA disagree significantly "
                f"(agreement={agreement:.0%}) — treat recommendation cautiously."
            )
        elif agreement >= AGREEMENT_HIGH_THRESHOLD:
            parts.append(f"Models agree strongly (agreement={agreement:.0%}).")

    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Single-row decision
# ---------------------------------------------------------------------------

def make_decision(row: pd.Series):

    pvi         = row["PVI"]
    viability   = row["viability"]
    growth_n    = row.get("growth_norm",    0.5)
    stability_n = row.get("stability_norm", 0.5)
    has_anomaly = row.get("has_anomaly",    False)
    anomaly_pct = row.get("anomaly_pct",    0.0)

    if growth_n >= GROWTH_POSITIVE_THRESHOLD:
        growth_label = "positive"
    elif growth_n <= GROWTH_NEGATIVE_THRESHOLD:
        growth_label = "declining"
    else:
        growth_label = "flat"

    if stability_n >= 0.60:
        risk_label = "low"
    elif stability_n >= STABILITY_LOW_THRESHOLD:
        risk_label = "moderate"
    else:
        risk_label = "high"

    is_positive_growth  = growth_n >= GROWTH_POSITIVE_THRESHOLD
    is_declining_growth = growth_n <= GROWTH_NEGATIVE_THRESHOLD
    is_high_risk        = stability_n < STABILITY_LOW_THRESHOLD
    serious_anomaly     = has_anomaly and anomaly_pct >= ANOMALY_SERIOUS_THRESHOLD

    anomaly_override = False

    if serious_anomaly:
        anomaly_override = True
        if is_declining_growth:
            decision = "Decrease"
        else:
            decision = "Hold"

    elif viability == "High":
        if is_positive_growth and not is_high_risk:
            decision = "Increase"
        else:
            decision = "Hold"

    elif viability == "Medium":
        if is_positive_growth and not is_high_risk:
            decision = "Increase"
        elif is_declining_growth and is_high_risk:
            decision = "Decrease"
        else:
            decision = "Hold"

    else:
        if is_declining_growth or is_high_risk:
            decision = "Decrease"
        elif is_positive_growth and not is_high_risk:
            decision = "Hold"
        else:
            decision = "Decrease"

    confidence  = compute_confidence(row)
    explanation = build_explanation(
        decision, row, growth_label, risk_label, anomaly_override
    )

    return decision, confidence, explanation


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(PVI_PATH):
        raise FileNotFoundError(
            f"PVI file not found at {PVI_PATH}.\n"
            "Run  python src/pvi.py  first."
        )

    pvi_df = pd.read_csv(PVI_PATH)
    print(f"Loaded PVI scores: {len(pvi_df)} items")

    results = []
    for _, row in pvi_df.iterrows():
        decision, confidence, explanation = make_decision(row)
        results.append({
            "store_id":        row["store_id"],
            "item_id":         row["item_id"],
            "cat_id":          row.get("cat_id",  ""),
            "dept_id":         row.get("dept_id", ""),
            "Decision":        decision,
            "Confidence":      confidence,
            "Explanation":     explanation,
            "PVI":             row["PVI"],
            "Viability":       row["viability"],
            "demand_norm":     round(row.get("demand_norm",    np.nan), 3),
            "growth_norm":     round(row.get("growth_norm",    np.nan), 3),
            "stability_norm":  round(row.get("stability_norm", np.nan), 3),
            "price_norm":      round(row.get("price_norm",     np.nan), 3),
            "forecast_mean":   round(row.get("forecast_mean",  np.nan), 2),
            "model_agreement": round(row.get("model_agreement", np.nan), 3)
                               if pd.notna(row.get("model_agreement")) else None,
            "has_anomaly":     row.get("has_anomaly",   False),
            "anomaly_count":   row.get("anomaly_count", 0),
            "anomaly_pct":     row.get("anomaly_pct",   0.0),
        })

    rec_df = pd.DataFrame(results)

    decision_order = {"Decrease": 0, "Increase": 1, "Hold": 2}
    rec_df["_sort_decision"] = rec_df["Decision"].map(decision_order)
    rec_df = rec_df.sort_values(
        ["_sort_decision", "store_id", "PVI"],
        ascending=[True, True, False],
    ).drop(columns=["_sort_decision"]).reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    rec_df.to_csv(OUTPUT_PATH, index=False)

    decision_counts    = rec_df["Decision"].value_counts()
    confidence_counts  = rec_df["Confidence"].value_counts()
    anomaly_count      = rec_df["has_anomaly"].sum()

    print(f"\n{'='*55}")
    print("Stock recommendations generated.")
    print(f"  Total items      : {len(rec_df)}")
    print(f"  Increase         : {decision_counts.get('Increase', 0)}")
    print(f"  Hold             : {decision_counts.get('Hold',     0)}")
    print(f"  Decrease         : {decision_counts.get('Decrease', 0)}")
    print(f"  High confidence  : {confidence_counts.get('High',   0)}")
    print(f"  Low confidence   : {confidence_counts.get('Low',    0)}")
    print(f"  Anomaly flagged  : {anomaly_count}")
    print(f"  Saved to         : {OUTPUT_PATH}")
    print(f"{'='*55}\n")

    print("Priority actions (Decrease / urgent items):")
    urgent = rec_df[rec_df["Decision"] == "Decrease"][
        ["store_id", "item_id", "cat_id", "PVI", "Viability",
         "Confidence", "has_anomaly"]
    ].head(10)
    print(urgent.to_string(index=False))

    return rec_df


if __name__ == "__main__":
    main()