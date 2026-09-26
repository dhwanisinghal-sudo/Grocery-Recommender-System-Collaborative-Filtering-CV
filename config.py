"""
config.py — Central configuration for the Smart Grocery Recommender System.

This replaces the config.py deleted on 2026-07-09. That version described the
retired Instacart / MobileNetV2 phase (orders.csv, MAX_USERS=5000, k-NN weights
SVD_WEIGHT=0.7 / ITEM_ITEM_WEIGHT=0.3) and was never imported anywhere in the
codebase — it drifted out of sync with the actual hybrid CF + vision-cascade
system silently, because nothing depended on it, so nothing broke when it went
stale. It was correctly deleted as dead weight.

This file exists so that CANNOT happen again: every hyperparameter here is
imported by BOTH app.py (the live app) and every script in experiments/ (the
verification/ablation/paper-table scripts). There is exactly one place these
numbers live. If you change a weight, both the app and the verification
scripts pick it up automatically — they cannot silently disagree.

Do not hardcode any of these numbers directly in app.py or experiments/*.py.
Import them from here instead.
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
