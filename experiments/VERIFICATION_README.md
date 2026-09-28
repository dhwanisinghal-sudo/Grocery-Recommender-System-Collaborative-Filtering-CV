# Verification Scripts

These scripts reproduce the numbers in my IEEE-format paper (Sections
VIII-B, VIII-C, VIII-E) and the vision-pipeline accuracy claim (§4.1 of
`docs/PROJECT_REPORT.md`). They run on the repo's actual data files. The
baseline metrics (Section VIII-A) come from running the deployed app itself,
through `compute_eval_metrics()` in `app.py`.

**Status:** ablation, K-sensitivity and grid search are also wired into the
app. `app.py`'s "📊 Evaluation Metrics" mode has four tabs (Baseline,
Ablation, K-Sensitivity, α/β Grid Search), backed by
`compute_ablation_metrics()`, `compute_k_sensitivity()` and
`compute_alpha_beta_grid()`. Those functions use the same split, seed and
sample logic as the scripts below, and I checked that they give the same
numbers as the standalone scripts (same RMSE, Precision/Recall/F1 and
grid-search ranking). The scripts still work on their own, which is handy if
you don't want to start Streamlit. There is no bootstrap confidence interval
anywhere in this codebase, only point estimates.

## Files

| Script | Paper section | What it reproduces |
|---|---|---|
| `verify_ablation.py` | VIII-B | Table IV: zero-fill vs. mean-centered SVD (RMSE, Precision/Recall/F1@10, coverage) |
| `verify_k_sensitivity.py` | VIII-C | Table V: Precision/Recall/F1 at K ∈ {5, 10, 20} |
| `verify_worked_example.py` | VIII-E | Table VI: top-5 recommendations per model and for the hybrid, for user U097 |
| `verify_grid_search.py` | not in the current paper | Grid search over the hybrid weights α, β (see `docs/PROJECT_REPORT.md` §4.2) |
| `verify_full_user_ci.py` | not in the current paper | All 149 evaluable users instead of the 50-user sample: P/R/F1 for User-CF, Item-CF, SVD and Hybrid with 95% bootstrap CIs, paired hybrid-vs-baseline comparison, and Hybrid variance over 10 split seeds (see `docs/PROJECT_REPORT.md` §5.1). Imports functions from `verify_grid_search.py`. |
| `verify_grid_full_users.py` | not in the current paper | The α/β grid re-run on all 149 users; shows the default still ranks first but the grid is nearly flat. |
| `verify_vision_accuracy.py` | not in the current paper | Per-stage vision-pipeline accuracy on the 260 labeled images in `data/vision_test_set/` (see `docs/PROJECT_REPORT.md` §4.1). Category-level accuracy, not per-SKU. Writes `vision_accuracy_results.csv`. Needs `pytesseract` and Tesseract installed. It calls the real Gemini/HF APIs if `GEMINI_API_KEY` / `HF_API_TOKEN` are set. Without a Gemini key it falls through to HF and then the color heuristic; without an HF token it skips HF. See the notes below for the two runs I have on record. |

Section VIII-D (Precision/Recall/F1 for heavy vs. light raters) isn't a
separate script. It's the same evaluation loop as `verify_k_sensitivity.py`,
filtered by median rating count, and I haven't built it as a script.

## Requirements

Same as `app.py`, without Streamlit:

```
pandas
numpy
scikit-learn
scipy
```

If you've already installed `requirements.txt` for the app, you have all of
it.

## Running

From the repo root:

```bash
python experiments/verify_ablation.py
python experiments/verify_k_sensitivity.py
python experiments/verify_worked_example.py
python experiments/verify_grid_search.py
python experiments/verify_vision_accuracy.py
```

The first four read `data/user_ratings.csv` and, where needed,
`data/products_500plus.csv`, the same files the app reads. Each prints its
output next to the paper's reference values so you can compare by eye. They
copy the model logic (`user_based_recommend`, `item_based_recommend`,
`svd_recommend`, `hybrid_recommend` and `compute_eval_metrics`) from `app.py`
instead of importing it, because `app.py` is a Streamlit script that runs UI
code at import time.

## Notes on reproducibility

- `verify_ablation.py` and `verify_k_sensitivity.py` use the same
  `train_test_split(..., test_size=0.2, random_state=42)` as
  `compute_eval_metrics()`, so the results are deterministic.
- `verify_worked_example.py` builds the models on the full ratings matrix
  with no split, which is how the app generates recommendations for a real
  user in the UI.
- `verify_grid_search.py` uses the same 80/20 split (seed 42) and fits the
  user-based, item-based and SVD models on the train split only. Fitting on
  the full matrix would leak test ratings into the similarity matrices.
- `verify_ablation.py`, `verify_k_sensitivity.py` and
  `verify_grid_search.py` evaluate Precision/Recall/F1/Coverage on a fixed
  random sample of 50 users (`random.Random(42).sample(...)`), the same as
  `compute_eval_metrics()` in `app.py`. I changed this from an earlier version
  that took the first 50 users in groupby order, which wasn't random. If you
  have numbers from before that change, expect a small shift. The current
  values are the correct ones.
- I ran all of them against the current `data/` files, after the sampling
  change. The reference tables printed next to each script's output are the
  original ones from the paper, which were made before the change. I kept them
  as a baseline for comparison. They are not the expected output any more.
- All results are point estimates. None of the scripts report a confidence
  interval.
- `verify_vision_accuracy.py` is the one script that imports from `app.py`
  (`app.classify_image()` and `app.find_products_from_tags()`) instead of
  copying the logic, so its numbers come from the vision cascade the app
  actually runs. It measures category-level accuracy (13 categories), not
  per-SKU accuracy against the 500-product catalog. The script's docstring and
  `docs/PROJECT_REPORT.md` §4.1 and §8 explain why.
- I have two `verify_vision_accuracy.py` runs on record. The first was without
  `GEMINI_API_KEY`: OCR, HF and the color fallback ran, and the overall
  accuracy was 51.5% (HF resolved 99 images at 24.2%, color fallback resolved
  33 at 6.1%). The second had both keys: OCR, Gemini and HF ran, overall
  accuracy was 87.7%, and the color fallback never triggered. The 87.7% run is
  the reference value in `docs/PROJECT_REPORT.md` §4.1, and
  `vision_accuracy_results.csv` is from that run. The CSV from the first run is
  in git history (commit `84131e0`).

## About the paper

The "Paper section" labels above refer to a manuscript draft that is **not
included in this repository**. The reference values from it are hard-coded in
the comparison block at the end of each script, so the tables can be checked
without the paper.
