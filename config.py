"""
config.py — Reference configuration for the Smart Grocery Recommender System.

This replaces the config.py deleted on 2026-07-09. That version described the
retired Instacart / MobileNetV2 phase (orders.csv, MAX_USERS=5000, k-NN weights
SVD_WEIGHT=0.7 / ITEM_ITEM_WEIGHT=0.3) and was never imported anywhere in the
codebase, so it drifted out of sync silently.

What actually depends on this file today:
  - Imported by: experiments/verify_ablation.py,
    experiments/verify_k_sensitivity.py, experiments/verify_worked_example.py.
  - NOT imported by: app.py, experiments/verify_grid_search.py,
    experiments/verify_vision_accuracy.py, or data/generate_*.py. Those keep
    their own inline copies of the same values (app.py hardcodes them,
    including the evaluation-tab functions ported from the scripts).

Every value below currently matches the value hardcoded in app.py and in
those scripts. That is maintained by hand, not enforced by code: if you change
a number here you must also change it in app.py and in the scripts that do not
import this file, or they will silently disagree. Wiring app.py (and the
remaining scripts) to import from here is not done yet.

New code should import from this file rather than hardcode these numbers.
"""

import os

# ===========================
# DATA PATHS
# ===========================
DATA_DIR      = "data"
PRODUCTS_PATH = os.path.join(DATA_DIR, "products_500plus.csv")
RATINGS_PATH  = os.path.join(DATA_DIR, "user_ratings.csv")
USERS_PATH    = os.path.join(DATA_DIR, "users_new.csv")

# ===========================
# HYBRID MODEL WEIGHTS
# (rank-reciprocal fusion: score = ALPHA/rank_ub + BETA/rank_ib + GAMMA/rank_svd)
# ===========================
HYBRID_ALPHA = 0.40   # user-based CF weight
HYBRID_BETA  = 0.35   # item-based CF weight
HYBRID_GAMMA = round(1.0 - HYBRID_ALPHA - HYBRID_BETA, 2)  # SVD weight, derived — never set independently

# ===========================
# COLLABORATIVE FILTERING PARAMETERS
# ===========================
USER_CF_N_NEIGHBORS = 15   # top-K similar users considered in user-based CF
SVD_K               = 20   # SVD latent factors (auto-capped to min(matrix shape) - 1 for small slices)
DEFAULT_TOP_N        = 10  # default recommendation list length shown in the UI

# ===========================
# EVALUATION PARAMETERS
# ===========================
EVAL_TEST_SIZE           = 0.2   # train/test split fraction
EVAL_RANDOM_STATE        = 42    # split seed
EVAL_K                   = 10    # K in Precision@K / Recall@K
EVAL_RELEVANCE_THRESHOLD = 3.5   # rating >= this counts as "relevant" for top-N eval
EVAL_SAMPLE_SIZE         = 50    # number of users sampled for Precision/Recall/Coverage
EVAL_SAMPLE_SEED         = 42    # seed for the random.Random(...).sample(...) draw

# ===========================
# VISION PIPELINE
# ===========================
# Stage order matters: each stage only runs if the previous one returns no
# usable tags. See app.py: classify_image().
VISION_STAGE_ORDER = ["ocr", "gemini", "huggingface", "color_fallback"]
VISION_CATEGORIES  = [
    "Personal Care", "Dairy", "Snacks", "Spices", "Drinks", "Health",
    "Home Care", "Grains", "Bakery", "Frozen", "Condiments", "Beverages",
    "Noodles",
]
