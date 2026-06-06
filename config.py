"""
config.py — Central configuration for Smart Grocery Recommender System
Edit paths and hyperparameters here without touching the notebook.
"""

import os

# ===========================
# DATA PATHS
# ===========================
DATA_DIR        = "data"
ORDERS_PATH     = os.path.join(DATA_DIR, "orders.csv")
PRIOR_PATH      = os.path.join(DATA_DIR, "order_products__prior.csv")
TRAIN_PATH      = os.path.join(DATA_DIR, "order_products__train.csv")
PRODUCTS_PATH   = os.path.join(DATA_DIR, "products.csv")

# ===========================
# MODEL SAVE PATHS
# ===========================
MODELS_DIR      = "models"
SVD_MODEL_PATH  = os.path.join(MODELS_DIR, "svd_model.pkl")
CV_MODEL_PATH   = os.path.join(MODELS_DIR, "mobilenetv2_grocery.h5")

# ===========================
# DATASET PARAMETERS
# ===========================
MAX_USERS       = 5000      # Number of users to use for CF training
RATING_SCALE    = (0, 10)   # Min/max rating for Surprise library
TEST_SIZE       = 0.2       # Train/test split ratio
RANDOM_STATE    = 42        # For reproducibility

# ===========================
# RECOMMENDATION PARAMETERS
# ===========================
TOP_N           = 5         # Number of recommendations to return
CANDIDATE_POOL  = 500       # Products to score per user

# ===========================
# COMPUTER VISION PARAMETERS
# ===========================
IMAGE_SIZE      = (224, 224)    # MobileNetV2 input size
CONFIDENCE_THRESHOLD = 0.5      # Min confidence to accept CV prediction

# ===========================
# HYBRID MODEL WEIGHTS
# ===========================
SVD_WEIGHT      = 0.7       # Weight for SVD score in hybrid model
ITEM_ITEM_WEIGHT = 0.3      # Weight for item-item similarity score
