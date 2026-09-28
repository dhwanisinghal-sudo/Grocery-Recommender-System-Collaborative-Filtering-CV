# Development Log

## About this log

I didn't keep a journal while building this. Most of my commit messages are
just "Update app.py" or "Add files via upload", so the git history on its own
doesn't say much. I put this log together afterwards from the history: which
files each commit touched, added, renamed or deleted, and in what order. When
I can't tell from a diff why I made a change, I only write what changed and
don't guess the reason. Where there are no commits, I say so.

To see the raw timeline:

```bash
git log --reverse --format="%ad|%s" --date=format:"%Y-%m-%d" --name-only
```

---

## Phase 1: Setting up the repo (Jun 5, 2026)

I created the repo with `README.md`, `requirements.txt`, `data/README.md` and
an early `PROJECT_REPORT.md`. The same day I uploaded my exploratory notebook,
`Smart_Grocery_Recommender_CV.ipynb` (the Instacart + MobileNetV2 phase, which
`PROJECT_REPORT.md` §9 describes as superseded).

## Phase 2: First version of the app (Jun 6–7, 2026)

I added `app.py` and `config.py`, and updated `.python-version` and
`requirements.txt` a few times while I sorted out dependencies. This is the
first time the Streamlit app exists separately from the notebook.

## Phase 3: Dataset and app iteration (Jun 9–19, 2026)

- **Jun 9:** I uploaded my first custom dataset (`data/products.csv`,
  `data/ratings.csv`, `data/users_new.csv`), which replaced the notebook's
  Instacart data for the app.
- **Jun 10:** a duplicate upload (`data/products (1).csv`) got created and I
  deleted it the same day, then re-uploaded `data/products.csv`.
- **Jun 10–19:** 55 commits to `app.py`, usually several a day.
- **Jun 19:** I deleted the early `data/products.csv` and `data/ratings.csv`
  and replaced them with `data/products_500plus.csv` and
  `data/user_ratings.csv`. These are the 500-product catalog and 6,796 ratings
  the app still uses (see `data/README.md`).

## Phase 4: More app work (Jun 28–29, 2026)

More `app.py` commits. No new files.

## Phase 5: Cleanup (Jul 9, 2026)

I deleted `config.py`. Its `SVD_WEIGHT` / `ITEM_ITEM_WEIGHT` values no longer
matched what `app.py` used and nothing imported it. I also removed the notebook
from the repo root and added `.gitignore`, `packages.txt` and
`secrets.toml.example` (the last one shows which Gemini/HF secrets the app
needs without committing real keys). `README.md` and `PROJECT_REPORT.md` were
updated to match.

## Phase 6: Verification scripts (Jul 10–22, 2026)

- **Jul 10:** `app.py` updates.
- **Jul 15–18:** `README.md` / `PROJECT_REPORT.md` updates.
- **Jul 17:** I added `experiments/verify_ablation.py`,
  `experiments/verify_k_sensitivity.py`,
  `experiments/verify_worked_example.py` and
  `experiments/VERIFICATION_README.md` in one commit.
- **Jul 22:** last `app.py` update of this stretch. I also removed a
  `.streamlit` directory (my local secrets config) from the repo.

## Gap: Jul 22 – Sep 11, 2026

There are no commits in this window, about seven weeks.

## Phase 7: Restructuring (Sep 11–15, 2026)

I updated `README.md`. On Sep 15 I put the notebook back under `notebooks/`
(I renamed it again on Sep 27, see Phase 9) and moved `PROJECT_REPORT.md` into
`docs/`.

## Phase 8: Verification and documentation pass (Sep 25–26, 2026)

- **Sep 25, 23:36–23:44:** added `experiments/verify_grid_search.py` and
  `experiments/grid_search_results.csv` (the α/β hybrid-weight grid search),
  and updated `VERIFICATION_README.md` and `docs/PROJECT_REPORT.md`.
- **Sep 26, 00:08–00:12:** changed `app.py`, `verify_ablation.py`,
  `verify_k_sensitivity.py` and `verify_grid_search.py` to draw the 50-user
  evaluation sample with `random.Random(42).sample(...)` instead of taking the
  first 50 users in groupby order. Regenerated the grid-search results and
  updated the docs.
- **Sep 26, 01:20–03:13:** added `data/generate_catalog.py` and
  `data/generate_ratings.py`, rewrote the root `README.md` for the current
  system, added the first version of this log, and updated `data/README.md`
  and `experiments/README.md`.
- **Sep 26, 21:49–22:19:** added `data/generate_users_missing.py` and updated
  `data/users_new.csv` (`data/README.md` notes that the file originally only
  covered U051–U150).

## Phase 9: Shared config, notebook archive, vision test set (Sep 27, 2026)

- **01:13–01:23:** documentation updates and two `app.py` edits.
- **01:38–01:42:** added `config.py` again and made `verify_ablation.py`,
  `verify_k_sensitivity.py` and `verify_worked_example.py` import it.
  `app.py` doesn't import it.
- **01:48–01:52:** renamed the notebook to `..._ARCHIVED.ipynb`, added
  `notebooks/README.md`, and updated `experiments/README.md`.
- **03:18–03:19:** updated `app.py` and added
  `experiments/verify_vision_accuracy.py`.
- **13:49–15:53:** uploaded the labeled vision test set (260 images, 20 per
  category across 13 categories) through the GitHub web UI, in roughly 50
  commits. The first `vision_accuracy_results.csv` was committed at 15:40.
- **16:00–17:15:** I found that `labels.csv` had ended up as a directory
  instead of a file, and fixed it (commit "Fix labels.csv directory/file
  collision; restore ground-truth CSV"). Updated `verify_vision_accuracy.py`.
- **18:32–18:33:** updated `README.md` and `docs/PROJECT_REPORT.md`.
- **19:13–19:14:** added a four-tab Evaluation Metrics mode to `app.py`
  (Baseline, Ablation, K-Sensitivity, α/β Grid Search), backed by
  `compute_ablation_metrics()`, `compute_k_sensitivity()` and
  `compute_alpha_beta_grid()`, and updated `VERIFICATION_README.md` to match.

## Phase 10: Regression, recovery and the Gemini run (Sep 27 23:59 – Sep 28, 2026)

- **Sep 27, 23:59 and Sep 28, 00:34:** two `app.py` commits. The first added an
  `import config`. After the second (338 lines removed), the evaluation-tab
  functions and the `config` import were no longer in `app.py`, while
  `VERIFICATION_README.md` still described the tabs. The commit messages don't
  say why.
- **Sep 28, 01:05:** changed the Gemini model list in `app.py` to
  `gemini-3.6-flash` / `gemini-3.1-flash-lite`.
- **Sep 28, 01:53:** updated `app.py`. The evaluation tabs were back, on top of
  the Gemini change.
- **Sep 28, 01:56–02:16:** updated the docs for the second vision run. I
  deleted `verify_vision_accuracy.py` and re-added it as
  `verify_vision_accuracy_patched.py`, and replaced
  `vision_accuracy_results.csv` with the run that used a Gemini key (87.7%
  overall). I also added the Colab notebook I ran it from (I later renamed it
  `experiments/vision_accuracy_colab.ipynb`).
- **Sep 28, 11:28:** renamed the script back to `verify_vision_accuracy.py`.
- **Sep 28, 11:36–11:41:** wording fixes. I added the synthetic-data
  limitation to `data/README.md` and `docs/PROJECT_REPORT.md`, changed the
  metrics note in `app.py` so it no longer calls the ratings "real", and
  corrected the `config.py` docstring and `experiments/README.md` to say which
  scripts import `config.py`.

---

## Phase 11: Wording cleanup (Sep 28, 11:55–12:20)

I went back through the docs and comments and reworded them. The scripts only
changed in their docstrings and comments.

- **11:55–11:57:** reworded `docs/DEVELOPMENT_LOG.md`,
  `docs/PROJECT_REPORT.md` and `experiments/VERIFICATION_README.md`.
- **12:11–12:13:** `README.md` (the tech stack now lists only what
  `requirements.txt` has), a comment in `app.py`, and the `config.py`
  docstring. Then `docs/DEVELOPMENT_LOG.md` and `docs/PROJECT_REPORT.md` again.
- **12:14–12:20:** `experiments/README.md`,
  `experiments/VERIFICATION_README.md`, `experiments/verify_grid_search.py`,
  `experiments/verify_vision_accuracy.py`, `data/README.md`,
  `data/generate_catalog.py`, `data/generate_ratings.py`,
  `data/generate_users_missing.py` and `notebooks/README.md`. At 12:18 I added
  `experiments/vision_accuracy_colab.ipynb`, the renamed copy of the Colab
  notebook from Phase 10.

---
