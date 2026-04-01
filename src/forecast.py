"""
forecast.py — Unified pipeline runner.

Runs the full data → forecast pipeline in sequence:
    Step 1: Preprocess raw M5 data  (preprocess.py)
    Step 2: Train Prophet models    (train_prophet.py)
    Step 3: Train ARIMA models      (train_arima.py)

Usage examples:
    # Default — top 200 items, both models (recommended for first run)
    python src/forecast.py

    # Run only on specific stores or categories
    python src/forecast.py --stores CA_1 TX_1 --categories FOODS

    # Train only ARIMA, skipping preprocessing (data already ready)
    python src/forecast.py --model arima --skip-preprocess

    # Bigger run — top 500 items, Prophet only
    python src/forecast.py --top-items 500 --model prophet
"""

import argparse
import sys
import time


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------

def run_preprocess(stores, categories, top_items):
    from preprocess import preprocess_m5
    _banner("STEP 1 — Preprocessing M5 data")
    t0 = time.time()
    preprocess_m5(
        sample_stores=stores or None,
        sample_categories=categories or None,
        top_n_items=top_items if top_items > 0 else None,
    )
    _elapsed(t0)


def run_prophet():
    from train_prophet import train_prophet_models
    _banner("STEP 2 — Prophet forecasting")
    t0 = time.time()
    train_prophet_models()
    _elapsed(t0)


def run_arima():
    from train_arima import train_arima_models
    _banner("STEP 3 — ARIMA forecasting")
    t0 = time.time()
    train_arima_models()
    _elapsed(t0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _elapsed(t0: float):
    print(f"\n  ✓ Done in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Retail Demand AI — forecasting pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
After this pipeline completes, run these next:
  python src/pvi.py        — compute Product Viability Index
  python src/recommend.py  — generate stock recommendations
  python src/evaluate.py   — evaluate forecast accuracy
        """,
    )
    parser.add_argument(
        "--stores", nargs="+", default=None,
        help="Store IDs to include, e.g. CA_1 TX_2  (default: all stores)",
    )
    parser.add_argument(
        "--categories", nargs="+", default=None,
        choices=["FOODS", "HOBBIES", "HOUSEHOLD"],
        help="Category IDs to include (default: all categories)",
    )
    parser.add_argument(
        "--top-items", type=int, default=200,
        help="Keep top N items by total sales per filters (0 = all, default: 200)",
    )
    parser.add_argument(
        "--model", choices=["prophet", "arima", "both"], default="both",
        help="Which forecasting model(s) to train (default: both)",
    )
    parser.add_argument(
        "--skip-preprocess", action="store_true",
        help="Skip Step 1 if processed_m5.csv already exists",
    )

    args = parser.parse_args()
    pipeline_start = time.time()

    # ── Step 1 ─────────────────────────────────────────────
    if not args.skip_preprocess:
        run_preprocess(args.stores, args.categories, args.top_items)
    else:
        print("\n[Skipping preprocessing — using existing processed_m5.csv]")

    # ── Step 2 ─────────────────────────────────────────────
    if args.model in ("prophet", "both"):
        run_prophet()

    # ── Step 3 ─────────────────────────────────────────────
    if args.model in ("arima", "both"):
        run_arima()

    # ── Summary ────────────────────────────────────────────
    total_elapsed = time.time() - pipeline_start
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {total_elapsed:.1f}s")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("  python src/pvi.py        — compute Product Viability Index")
    print("  python src/recommend.py  — generate stock recommendations")
    print("  python src/evaluate.py   — evaluate forecast accuracy")
    print("  uvicorn app.api:app --reload  — start the API server\n")


if __name__ == "__main__":
    main()