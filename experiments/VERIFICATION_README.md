# Ablation & Worked-Example Verification Scripts

These three scripts let anyone reproduce the non-baseline numbers reported
in the IEEE paper (Sections VIII-B, VIII-C, VIII-E) by running real
computation against the repo's actual data files — no train-time secrets,
no hidden state. They complement, and do not replace, the baseline metrics
already reproducible by running the deployed app directly (Section VIII-A,
`compute_eval_metrics()` in `app.py`).

## Files

| Script | Paper section | Reproduces |
|---|---|---|
| `verify_ablation.py` | VIII-B | Table IV — zero-fill vs. mean-centered SVD (RMSE, Precision/Recall/F1@10, coverage) |
| `verify_k_sensitivity.py` | VIII-C | Table V — Precision/Recall/F1 at K ∈ {5, 10, 20} |
| `verify_worked_example.py` | VIII-E | Table VI — top-5 recommendations per model + hybrid, for user U097 |

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
```

Each script reads `data/user_ratings.csv` and (where relevant)
`data/products_500plus.csv` directly — the same files the deployed app
reads — and prints its own output side-by-side with the paper's reference
values for a quick visual diff. Nothing is imported from `app.py` itself
(that file is a Streamlit script with UI code baked in at import time);
instead, each script reimplements only the exact model logic
(`user_based_recommend`, `item_based_recommend`, `svd_recommend`,
`hybrid_recommend`, and `compute_eval_metrics`) verbatim from `app.py`, so
results are guaranteed to match the deployed app's behavior, not an
approximation of it.

## Notes on reproducibility

- `verify_ablation.py` and `verify_k_sensitivity.py` use the same
  `train_test_split(..., test_size=0.2, random_state=42)` as
  `compute_eval_metrics()`, so results are deterministic and match the
  paper exactly, not just approximately.
- `verify_worked_example.py` builds models on the full ratings matrix
  (no split), matching how the live app generates recommendations for a
  real user in the UI.
- All three were run against the current `data/` files at the time this
  README was written and reproduced the paper's tables exactly.
