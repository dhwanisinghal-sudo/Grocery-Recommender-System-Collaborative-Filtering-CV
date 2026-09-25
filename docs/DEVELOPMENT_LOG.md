# 🗓️ Development Log

## How this log was built

This repository's 150+ commits mostly carry generic messages ("Update
app.py", "Add files via upload") that don't say what changed or why. That's
a real gap on its own — commit history should be able to answer "what
happened and when" without someone having to `git diff` every commit.

This log closes that gap by reconstructing the actual timeline from what
each commit's **file-level diff** shows (which files were touched, added,
renamed, or deleted, in what order), cross-checked against the current
architecture described in `PROJECT_REPORT.md`. It does not invent
information that isn't recoverable from the repo itself — where a period
has no commits, that's stated plainly rather than papered over, and where
a commit's content doesn't clearly explain intent, this log describes only
what the diff shows, not a guessed rationale.

Regenerate the raw timeline yourself with:

```bash
git log --reverse --format="%ad|%s" --date=format:"%Y-%m-%d" --name-only
```

---

## Phase 1 — Project scaffolding (Jun 5, 2026)

Repo initialized with `README.md`, `requirements.txt`, `data/README.md`,
and an early `PROJECT_REPORT.md`. The original exploratory notebook,
`Smart_Grocery_Recommender_CV.ipynb` (the Instacart + MobileNetV2 phase
now described as superseded in `PROJECT_REPORT.md` §9), was uploaded the
same day — so the notebook-based CV/CF proof-of-concept was the starting
point, not an afterthought.

## Phase 2 — First deployed app skeleton (Jun 6–7, 2026)

`app.py` and a `config.py` created; `.python-version` and iterative
`requirements.txt` updates followed as dependencies were pinned down. This
is the first appearance of the Streamlit app as a separate artifact from
the notebook.

## Phase 3 — Early dataset + heavy app iteration (Jun 9–19, 2026)

- **Jun 9:** first custom dataset uploaded — `data/products.csv`,
  `data/ratings.csv`, `data/users_new.csv` — replacing reliance on the
  notebook's Instacart data for the app.
- **Jun 10:** a duplicate-upload mistake (`data/products (1).csv`) was
  created and deleted within the same day, then re-uploaded cleanly as
  `data/products.csv` — ordinary iteration noise, not a milestone, but
  visible in the diff.
- **Jun 10–19:** `app.py` received on the order of 60+ incremental commits
  in this window (multiple same-day commits most days), consistent with
  active, iterative development of the recommendation/scanning logic
  rather than one large drop.
- **Jun 19:** the early `data/products.csv` / `data/ratings.csv` were
  **deleted and replaced** with `data/products_500plus.csv` and
  `data/user_ratings.csv` — the 500-product catalog and 6,796-rating
  dataset that the deployed app still uses today. This is the point where
  the current dataset (see `data/README.md`) was adopted.

## Phase 4 — Continued app work (Jun 28–29, 2026)

Further `app.py` commits; no new files introduced in this window.

## Phase 5 — Cleanup and deployment hardening (Jul 9, 2026)

A dense, single-day cleanup: `config.py` deleted (its `SVD_WEIGHT` /
`ITEM_ITEM_WEIGHT` values had drifted out of sync with what `app.py`
actually used and were unused — `app.py`'s current hybrid-config comment
block documents this directly), the original exploratory notebook removed
from the repo root, and `.gitignore`, `packages.txt`, and
`secrets.toml.example` added — the last being Streamlit's convention for
documenting required secrets (Gemini/HF API keys) without committing real
ones. `README.md` and `PROJECT_REPORT.md` were updated to match.

## Phase 6 — Verification scripts introduced (Jul 10–22, 2026)

- **Jul 10:** further `app.py` updates.
- **Jul 15–18:** `README.md` / `PROJECT_REPORT.md` documentation passes.
- **Jul 17:** `experiments/verify_ablation.py`,
  `experiments/verify_k_sensitivity.py`,
  `experiments/verify_worked_example.py`, and
  `experiments/VERIFICATION_README.md` all added in one commit — the
  first version of the reproducibility scripts tied to the accompanying
  paper's ablation and K-sensitivity tables.
- **Jul 22:** final `app.py` update and removal of a `.streamlit`
  directory (local secrets config not meant to be committed) before the
  gap below.

## Gap: Jul 22 – Sep 11, 2026 (no commits)

No commits exist in this ~7-week window. This log states that plainly
rather than guessing at unlogged work — if asked, this is the honest
answer: the repository shows no activity here.

## Phase 7 — Repo restructuring (Sep 11–15, 2026)

`README.md` updated; then, on Sep 15, the repo was reorganized for
cleanliness: the exploratory notebook was restored under
`notebooks/Smart_Grocery_Recommender_(ML,CV).ipynb` (renamed and re-added
after its Jul 9 removal — this is the version referenced today in
`PROJECT_REPORT.md` §9), and `PROJECT_REPORT.md` was moved into `docs/`.

## Phase 8 — Gap-closing pass (Sep 25–26, 2026)

The most recent, best-documented phase, closing several previously-open
verification gaps:

- **Sep 25:** `experiments/verify_grid_search.py` and
  `experiments/grid_search_results.csv` added — the α/β hybrid-weight grid
  search; `experiments/VERIFICATION_README.md` and `docs/PROJECT_REPORT.md`
  updated to describe it.
- **Sep 26:** `app.py`, `verify_ablation.py`, and `verify_k_sensitivity.py`
  updated to draw their 50-user evaluation sample via
  `random.Random(42).sample(...)` instead of the first 50 users in
  groupby insertion order; `data/generate_catalog.py` and
  `data/generate_ratings.py` added to make the dataset's curation process
  reproducible; `README.md` rewritten to describe the current deployed
  system instead of the retired Instacart/MobileNetV2 notebook numbers;
  `docs/PROJECT_REPORT.md` and `experiments/VERIFICATION_README.md`
  updated throughout to match.

---

## What this log does and doesn't establish

**Does:** shows the project was built incrementally over roughly 3.5
months (Jun 5 – Sep 26, 2026) across at least 25 distinct working days,
with identifiable phases (prototype → dataset swap → cleanup →
verification tooling → restructuring → gap-closing), each backed by real
file-level changes rather than a single bulk commit.

**Doesn't:** explain *why* any individual line changed within a given
`app.py` commit (the messages don't say, and this log doesn't invent
that), and doesn't account for the Jul 22–Sep 11 gap beyond noting it
exists. If asked about either in a review, those are the honest answers.
