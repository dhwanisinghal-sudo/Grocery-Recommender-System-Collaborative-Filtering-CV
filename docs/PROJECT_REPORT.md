# 📋 Smart Grocery Recommender System — Project Report

**Domain:** Machine Learning + Computer Vision
**Application:** Multi-stage image-based product identification + hybrid collaborative-filtering recommendations
**Status:** ✅ Deployed — Streamlit application

> This report describes the **current, deployed system** (`app.py`). Full
> methodology and evaluation results also live in
> `Smart_Grocery_Recommender_Paper_Corrected.docx`; a user-facing summary is
> in [`README.md`](README.md). An earlier, superseded exploratory phase
> (Instacart dataset + MobileNetV2 classifier) is documented separately in
> **§9 — Earlier Exploratory Phase (Superseded)** below, for historical
> reference only.

---

## 1. Problem Statement

Traditional grocery apps recommend the same popular products to everyone — ignoring individual purchase history and the practical difficulty of identifying an item from a photo of its packaging. This project builds a **Smart Grocery Recommendation System** that:

- Identifies grocery products from real-world photos through a multi-stage computer-vision pipeline
- Recommends personalized products using a hybrid collaborative-filtering engine
- Combines both into an end-to-end scan → identify → recommend workflow, deployed as an interactive Streamlit app

---

## 2. Dataset

| Attribute            | Value      |
| --------------------- | ---------- |
| Products               | 500        |
| Categories               | 13         |
| Users                      | 150        |
| Ratings                       | 6,796      |
| Rating scale                     | 1.5 – 5.0  |
| Mean rating                          | 3.79       |
| Matrix sparsity                          | 90.9%      |
| Avg. ratings / user                          | 45.3       |

Catalog spans Personal Care, Dairy, Snacks, Spices, Drinks, Health, Home
Care, Grains, Bakery, Frozen, Condiments, Beverages, and Noodles, including
branded items (Amul, Parle, Britannia, MDH, Haldiram's, Patanjali, etc.).

Data files, under `data/`:

- `products_500plus.csv`
- `user_ratings.csv`

---

## 3. System Architecture

```
📷 Image Upload
      ↓
🔎 4-Stage Vision Pipeline
   1. OCR (Tesseract) — match on-package text to GROCERY_KEYWORDS
   2. Gemini Vision fallback — constrained tag vocabulary + confidence
   3. Hugging Face Inference fallback — ImageNet-style labels, normalized
   4. Color-heuristic fallback — hue/brightness/texture rule-based guess
      ↓
🗂️ Catalog Match (500-product Indian grocery catalog)
      ↓
🤖 Hybrid Collaborative Filtering
   User-based CF + Item-based CF + SVD (rank-reciprocal blend)
      ↓
✅ Personalized Recommendations
```

A `DAIRY_SPECIFIC` priority-ordered, mutually-exclusive tag structure
disambiguates visually similar dairy products (butter, ghee, paneer, curd,
cheese, cream, milk). A manual text-search override lets a user correct a
misclassification directly.

---

## 4. Models Used

### 4.1 Vision Pipeline

| Stage                 | Method                                                             | Role                                             |
| ---------------------- | -------------------------------------------------------------------- | -------------------------------------------------- |
| 1 — OCR                    | Tesseract, matched against 150+ keyword dictionary                       | Highest priority, tried first                          |
| 2 — Gemini Vision              | Google Gemini API, constrained tag + confidence output                       | Fallback if OCR is inconclusive                            |
| 3 — Hugging Face                    | General-purpose vision classifier, normalized onto product-tag vocabulary        | Fallback if Gemini is unavailable/inconclusive                  |
| 4 — Color heuristic                       | Rule-based hue/brightness/texture classifier                                         | Final fallback if both API stages are unavailable                     |

The vision pipeline has **not been formally accuracy-evaluated** — no
labeled image test set currently exists for this catalog (see §8,
Limitations).

### 4.2 Collaborative Filtering Models

| Model                     | Method                                                                                                                |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| User-based CF                  | Cosine similarity between user rating vectors; 15 nearest neighbors aggregated over unrated items                            |
| Item-based CF                       | Cosine similarity between item vectors; restricted to `RELATED_CATEGORIES` for post-scan suggestions                             |
| SVD                                       | Truncated SVD, k=20 latent factors, via `scipy.sparse.linalg.svds`, reconstructed to a dense predicted-rating matrix                  |

The three ranked lists are combined via a rank-reciprocal hybrid score:

```
score(item) = α · (1 / rank_user_based)
            + β · (1 / rank_item_based)
            + (1 − α − β) · (1 / rank_svd)
```

Default weights: `α = 0.40`, `β = 0.35` (fixed defaults, adjustable at
runtime, not learned or grid-searched). New users with no rating history
get popularity-based recommendations (interaction count × average rating),
filtered to preferred categories.

---

## 5. Evaluation Metrics

Computed via `compute_eval_metrics()` in `app.py`, on an 80/20 train-test
split (seed=42) of the 6,796 ratings.

| Metric | SVD (zero-fill) | User-Based CF |
| ------- | ----------------- | --------------- |
| RMSE      | 3.61                 | 0.88            |

| Metric                | Value (K=10) |
| ------------------------ | -------------- |
| Precision@10                | 3.4%           |
| Recall@10                       | 3.7%           |
| F1                                    | 3.5%           |
| Catalog Coverage                          | 48.2%          |

Precision@10 stays low mainly because the dataset is small and 90.9%
sparse — not because the underlying models are broken. Additional
ablations (mean-centered SVD, K-sensitivity, activity-level breakdown) are
reported in the paper but currently live in standalone analysis scripts,
not the deployed evaluation dashboard.

---

## 6. App Modes

The app is a single-page Streamlit application (`app.py`, ~1,700 lines)
with seven sidebar modes:

- User recommendations (hybrid CF)
- Similar-product lookup (item-based CF)
- Image-based scanning
- Cold-start recommendations for new users
- Evaluation-metrics dashboard
- Catalog / user search
- Raw data explorer (Products, Ratings, Insights tabs)

---

## 7. Tech Stack

| Category                     | Libraries                                                              |
| ------------------------------ | -------------------------------------------------------------------------- |
| Language                          | Python 3.x                                                                     |
| Frontend                              | Streamlit                                                                          |
| Collaborative Filtering                   | Surprise (User-Based CF, Item-Based CF, SVD), scikit-learn, Pandas, NumPy               |
| Computer Vision & OCR                         | OpenCV, Tesseract OCR, Pillow (PIL)                                                         |
| AI Models                                         | Google Gemini API, Hugging Face Transformers                                                    |
| Visualization                                         | Matplotlib, Plotly                                                                                  |
| Version Control                                           | Git, GitHub                                                                                             |

---

## 8. Results Summary & Limitations

**Results:**

- The vision pipeline resolves a product photo to a catalog item through
  up to four fallback stages, favoring precision (OCR / constrained-tag
  matching) over the more permissive final heuristic stage.
- User-Based CF substantially outperforms zero-fill SVD on this dataset
  (RMSE 0.88 vs 3.61), reflecting the benefit of neighborhood-based
  methods on small, sparse rating matrices.
- Data loading and all three CF models are cached (`@st.cache_data`, keyed
  on ratings-table length), so models are computed once per data version
  rather than recomputed on every interaction.

**Known gaps:**

- The vision pipeline has no labeled image test set, so end-to-end
  identification accuracy is not currently quantified.
- Hybrid CF weights (α, β) are fixed defaults rather than tuned or
  grid-searched.
- Mean-centered SVD, K-sensitivity, and activity-level ablations exist as
  standalone scripts, not yet wired into the deployed evaluation dashboard.

---

## 9. Earlier Exploratory Phase (Superseded)

> ⚠️ **Historical only.** Everything in this section describes a retired,
> architecturally unrelated notebook (`Smart_Grocery_Recommender_CV.ipynb`),
> not the deployed app described in §1–8 above. It predates and was
> superseded by that system.

The project's first exploratory phase used the public **Instacart Market
Basket Analysis** dataset with a generic **MobileNetV2** ImageNet
classifier, run as a Google Colab notebook, before moving to a curated,
deployable catalog and a purpose-built vision pipeline.

**Dataset (notebook phase):**

| Attribute            | Value                                                                                    |
| ---------------------- | -------------------------------------------------------------------------------------------- |
| Source                    | [Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis)   |
| Total Orders                  | 3,421,083                                                                                            |
| Total Products                    | 49,688                                                                                                   |
| Total Users                           | 206,209                                                                                                      |
| Users Used                                | 4,628 (user_id ≤ 5000)                                                                                          |

**Models (notebook phase):**

| Model                          | RMSE / Accuracy    | Notes                              |
| --------------------------------- | ---------------------- | -------------------------------------- |
| SVD                                    | RMSE 1.7034              | Best single CF model                       |
| KNNBasic                                   | RMSE 2.1500                  | User-user similarity                           |
| NMF                                            | RMSE 1.9200                      | Matrix factorization                                |
| Hybrid (SVD + KNN)                                 | RMSE 1.6800                          | Best overall CF model                                   |
| MobileNetV2 (ImageNet)                                 | 92.34% accuracy                          | Grocery image classification                                |

**Why the project moved on:** this notebook validated the CV + CF concept
at scale, but relied on a large third-party dataset not tied to a specific
deployable catalog, and a generic ImageNet classifier not tuned to any
particular product set or packaging. The project then moved to the curated
500-product catalog and multi-stage OCR/Gemini/Hugging Face vision
pipeline described in §1–8, which is better suited to real product-package
recognition and to deployment as an interactive app.

The notebook is retained in the repository for reference only and is not
maintained or evaluated against current app data.
