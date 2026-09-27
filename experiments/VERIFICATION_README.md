# Verification Scripts

These five scripts let anyone reproduce the non-baseline numbers reported
in the IEEE paper (Sections VIII-B, VIII-C, VIII-E) and the vision-pipeline
accuracy claim (§4.1 of `docs/PROJECT_REPORT.md`) by running real
computation against the repo's actual data files — no train-time secrets,
no hidden state. They complement, and do not replace, the baseline metrics
already reproducible by running the deployed app directly (Section VIII-A,
`compute_eval_metrics()` in `app.py`).

**Status:** all three (ablation, K-sensitivity, grid search) are now also
wired directly into the deployed app — `app.py`'s "📊 Evaluation Metrics"
mode has four tabs (Baseline, Ablation, K-Sensitivity, α/β Grid Search),
backed by `compute_ablation_metrics()`, `compute_k_sensitivity()`, and
`compute_alpha_beta_grid()`, which reimplement the exact same
split/seed/sample logic as the scripts below — verified to produce
identical numbers to the standalone scripts (same RMSE, Precision/Recall/
F1, and grid-search ranking, to the decimal place). These CLI scripts
still exist and still work standalone (useful for offline/CI reproduction
or anyone who doesn't want to spin up Streamlit); the app's tabs are for
seeing the same tables without a terminal. There is still no bootstrap
confidence interval anywhere in this codebase — only point estimates.

## Files

| Script | Paper section | Reproduces |
|---|---|---|
| `verify_ablation.py` | VIII-B | Table IV — zero-fill vs. mean-centered SVD (RMSE, Precision/Recall/F1@10, coverage) |
| `verify_k_sensitivity.py` | VIII-C | Table V — Precision/Recall/F1 at K ∈ {5, 10, 20} |
| `verify_worked_example.py` | VIII-E | Table VI — top-5 recommendations per model + hybrid, for user U097 |
| `verify_grid_search.py` | — (not in current paper) | Grid search over hybrid weights α, β — see `docs/PROJECT_REPORT.md` §4.2 |
| `verify_vision_accuracy.py` | — (not in current paper) | Per-stage vision-pipeline accuracy on `data/vision_test_set/` (260 labeled images) — see `docs/PROJECT_REPORT.md` §4.1. Category-level accuracy, not per-SKU; writes `vision_accuracy_results.csv`. Requires `pytesseract` + Tesseract installed; hits the real Gemini/HF APIs if `GEMINI_API_KEY`/`HF_API_TOKEN` are configured (falls back to OCR/color-only otherwise — see notes below on the two runs on record). |

Section VIII-D (breakdown by user activity level, Table V-D... actually
Table labeled "heavy vs. light raters") is a straightforward re-slice of
the same `verify_k_sensitivity.py`-style predictions by median rating
count and isn't included as a separate script; it's a filtered rerun of
the same evaluation loop, not a distinct computation.

## Requirements

Same dependencies as `app.py`, minus Streamlit-specific ones:

```
pandas
numpy
scikit-learn
scipy
```

If you already have `requirements.txt` installed for the app, you have
everything needed.

## Running

From the repo root:

```bash
python experiments/verify_ablation.py
python experiments/verify_k_sensitivity.py
python experiments/verify_worked_example.py
python experiments/verify_grid_search.py
python experiments/verify_vision_accuracy.py
```

The first four read `data/user_ratings.csv` and (where relevant)
`data/products_500plus.csv` directly — the same files the deployed app
reads — and print their own output side-by-side with the paper's reference
values for a quick visual diff. Each of those four reimplements only the
exact model logic (`user_based_recommend`, `item_based_recommend`,
`svd_recommend`, `hybrid_recommend`, and `compute_eval_metrics`) verbatim
from `app.py`, rather than importing `app.py` itself (a Streamlit script
with UI code baked in at import time), so results are guaranteed to match
the deployed app's behavior, not an
approximation of it.

## Notes on reproducibility

- `verify_ablation.py` and `verify_k_sensitivity.py` use the same
  `train_test_split(..., test_size=0.2, random_state=42)` as
  `compute_eval_metrics()`, so results are deterministic and match the
  paper exactly, not just approximately.
- `verify_worked_example.py` builds models on the full ratings matrix
  (no split), matching how the live app generates recommendations for a
  real user in the UI.
- `verify_grid_search.py` also uses the same 80/20 split (seed=42) and
  fits the user-based/item-based/SVD models on the train split only
  (fitting on the full matrix would leak test ratings into the similarity
  matrices).
- `verify_ablation.py`, `verify_k_sensitivity.py`, and
  `verify_grid_search.py` all evaluate Precision/Recall/F1/Coverage on a
  **fixed random sample of 50 users** (`random.Random(42).sample(...)`),
  matching `compute_eval_metrics()` in `app.py`. This replaced an earlier
  methodology that took the first 50 users in groupby insertion order
  (not random) — if you see numbers from before that fix, expect a small
  shift; these are the current, correct values.
- All were run against the current `data/` files at the time this README
  was written and reflect the sample-fix above; the IEEE paper's original
  reference tables (printed alongside each script's output) predate the
  fix and are kept as a historical baseline for comparison, not as the
  current expected output.
- Point estimates only — none of these scripts currently report a
  confidence interval alongside Precision/Recall/F1.
- `verify_vision_accuracy.py` is the exception to the "reimplements rather
  than imports" rule above: it imports `app.classify_image()` and
  `app.find_products_from_tags()` directly, so its numbers are guaranteed
  to reflect the actual deployed vision cascade, not a reimplementation
  that could drift out of sync. It measures **category-level** accuracy
  (13 categories), not per-SKU accuracy against the 500-product catalog —
  see its own docstring and `docs/PROJECT_REPORT.md` §4.1/§8 for why.
- Two `verify_vision_accuracy.py` runs are on record: an initial run with
  no `GEMINI_API_KEY`/`HF_API_TOKEN` set (OCR + color-fallback only,
  51.5% overall), and a follow-up run with real keys configured (OCR +
  Gemini + HF, 87.7% overall, color fallback never triggered). The
  87.7% run is the current reference value in `docs/PROJECT_REPORT.md`
  §4.1; `vision_accuracy_results.csv` reflects that run.
