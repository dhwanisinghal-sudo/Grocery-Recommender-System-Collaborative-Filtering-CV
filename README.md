# 🛒 Smart Grocery Recommender System

### Hybrid Collaborative Filtering + Multi-Stage Computer Vision

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-CF-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

> This project is written up as a short conference-style paper, *"Smart Grocery Recommender: A Hybrid Collaborative Filtering System with a Multi-Stage Vision Pipeline for Personalized Grocery Recommendations"* (Dhwani Singhal). This README summarizes it; see `PROJECT_REPORT.md` for the fuller write-up including the ablation study and worked example.

---

## 📌 Project Overview

A **Smart Grocery Recommendation System**, deployed as a Streamlit app, that combines a **hybrid collaborative-filtering engine** with a **four-stage computer-vision identification pipeline** over a curated 500-product Indian grocery catalog. The system addresses two discovery problems at once: generic, non-personalized recommendations, and the inability to identify a product from a photo when its exact catalog name/brand spelling isn't known.

**Users can:**
- 📷 Upload a photo of a grocery item and have it identified via OCR → Gemini Vision → HuggingFace → color-heuristic fallback
- 🛒 Get personalized recommendations via a tunable hybrid CF model (user-based + item-based + SVD)
- 🔁 See "you might also like" suggestions restricted to plausibly-related categories
- 📊 Run live evaluation metrics against the real rating data
- 🔎 Search the catalog and browse raw data

> ⚠️ **Note:** this repo also contains `Smart_Grocery_Recommender_CV.ipynb`, an early exploratory notebook built on the Instacart dataset with a stock MobileNetV2 classifier. It is architecturally distinct from — and superseded by — the deployed app below. See [Notebook vs. App](#-notebook-vs-app).

---

## 📦 Dataset

| Attribute | Value |
|---|---|
| Source | Curated, synthetic Indian grocery catalog (Amul, Parle, Britannia, MDH, Haldiram's, Patanjali, and others) |
| Products | 500 |
| Categories | 13 — Personal Care, Dairy, Snacks, Spices, Drinks, Health, Home Care, Grains, Bakery, Frozen, Condiments, Beverages, Noodles |
| Users | 150 |
| Ratings | 6,796 explicit ratings |
| Rating scale | 1.5 – 5.0 |
| Mean rating | 3.79 |
| Matrix sparsity | 90.9% |
| Avg. ratings / user | 45.3 |
| Avg. ratings / product | 13.6 |

---

## 🔁 Pipeline

**1. Image Recognition (4-stage fallback chain)** — each stage runs only if the previous one fails to produce a confident match:

```
📷 Image Upload
      ↓
📝 OCR (Tesseract) ── matches on-package text against 150+ brand/product keywords
      ↓ (no confident match)
✨ Gemini Vision ── vision-language model, closed tag vocabulary, structured JSON output
      ↓ (unavailable / fails)
🤗 HuggingFace Inference ── general-purpose classifier, labels normalized to the same vocabulary
      ↓ (unavailable / fails)
🎨 Color-Heuristic Fallback ── hue/brightness/texture rule-based guess
      ↓
🗂️ Product Catalog Match (+ manual search override in the UI)
```

Packaging text is largely invariant to lighting and camera angle, making it a more reliable identity signal than appearance alone for packaged goods — so OCR is tried first, with vision-model and heuristic stages reserved for cases where text is absent, occluded, or illegible (e.g. loose produce).

**2. Recommendation (Hybrid Collaborative Filtering)** — three signals, computed independently and combined:

```
🤖 User-Based CF (cosine similarity, 15 nearest neighbors)   — weight α = 0.40
🤖 Item-Item CF (cosine similarity over item vectors)        — weight β = 0.35
🤖 SVD (scipy.sparse.linalg.svds, k=20 latent factors)       — weight γ = 1 − α − β = 0.25
      ↓
Rank-reciprocal fusion: score(item) += weight × 1 / (rank + 1)
      ↓
✅ Top-N Personalized Recommendations
```

α and β are adjustable live via sidebar sliders. New users with no rating history fall back to popularity-based recommendations (interaction count × average rating), filtered to their selected categories.

---

## 📊 Evaluation

Evaluated on an 80/20 train-test split (seed=42) of the 6,796 ratings. RMSE is computed on the intersection of test users/products present in the fitted prediction matrix; ranking metrics use a 50-user sample and a relevance threshold of rating ≥ 3.5.

**Baseline (zero-fill SVD, as implemented in the deployed app's `compute_eval_metrics()`):**

| Metric | SVD (zero-fill) | User-Based CF |
|--------|------------------|---------------|
| RMSE | 3.6145 | 0.8820 |

| Metric | Value |
|--------|-------|
| Precision@10 | 3.40% |
| Recall@10 | 3.66% |
| F1 Score | 3.53% |
| Catalog Coverage | 48.2% |

**Key finding — the SVD RMSE gap is a zero-fill artifact, not a model-quality difference.** Treating unrated entries as "rated zero" biases factorization toward under-prediction. Re-running SVD after mean-centering each user's ratings (subtracting their mean before factorization, adding it back after, clipped to [1.5, 5.0]) closes the RMSE gap entirely and improves ranking quality:

| Metric | Zero-Fill SVD | Mean-Centered SVD |
|--------|---------------|--------------------|
| RMSE | 3.6145 | **0.8789** (≈ matches User-Based CF's 0.8820) |
| Precision@10 | 3.40% | 3.40% |
| Recall@10 | 3.66% | **5.51%** (+49%) |
| F1 Score | 3.53% | **4.21%** (+19%) |
| Coverage | 48.2% | 40.6% |

Precision stays flat and coverage drops modestly, indicating mean-centering sharpens rather than broadens SVD's confident predictions. **Note:** this mean-centered variant is a reported ablation/analysis from the paper, not (yet) the code path executing in the deployed app's `compute_eval_metrics()`, which currently reports the zero-fill numbers above. See `PROJECT_REPORT.md` for the full ablation write-up, K-sensitivity analysis (K ∈ {5, 10, 20}), and a breakdown by user activity level.

Precision@10 remains low (3.4%) primarily because of dataset scale — 150 users and 90.9% sparsity leave limited overlap between predicted top-10 lists and held-out relevant items, and a 3.5 threshold captures most observed ratings (mean 3.79), enlarging the relevant-item set relative to any fixed K. This is a data-scale limitation, not a sign the models are behaving incorrectly — see the worked example in `PROJECT_REPORT.md`.

---

## ✅ Features

| Feature | Description |
|---------|-------------|
| 🔍 Multi-stage Image Recognition | OCR → Gemini Vision → HuggingFace → color fallback, plus manual search override |
| 🤝 Tunable Hybrid CF | User-CF + Item-CF + SVD, weights adjustable via sidebar sliders |
| 👤 Personalized | Recommendations based on individual rating history |
| ❄️ Cold Start | Popularity-based recommendations for new users, by selected category |
| 🔁 Cross-Category "You Might Also Like" | `RELATED_CATEGORIES`-driven item-based suggestions after an image match |
| 📊 Live Evaluation | RMSE, Precision@K, Recall@K, F1, Coverage computed on demand |
| 🔎 Search | Product and user search |
| 📋 Data Explorer | Browse products, ratings, and dataset insights |

Seven interactive modes in total: recommendations, similar-product lookup, image scan, cold start, evaluation metrics, search, and data explorer.

---

## 🛠️ Tech Stack

| Category | Libraries |
|----------|-----------|
| App Framework | Streamlit (single-file app, `@st.cache_data` keyed on ratings-table length) |
| Collaborative Filtering | scikit-learn (cosine similarity), scipy (`svds`) |
| Image Recognition | pytesseract (OCR), Gemini Vision API, HuggingFace Inference API |
| Image Processing | Pillow |
| Data Processing | Pandas, NumPy |
| HTTP | requests |

See [`requirements.txt`](requirements.txt) for exact packages, and note the system-level `tesseract-ocr` binary requirement below.

---

## 🚀 Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

OCR requires the **Tesseract OCR engine** as a system binary (pip alone won't install this):

```bash
# Debian/Ubuntu
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

For Streamlit Community Cloud, this repo includes a `packages.txt` (containing `tesseract-ocr`) which Streamlit Cloud installs automatically at deploy time.

### 2. Configure API keys

The app uses `st.secrets["GEMINI_API_KEY"]` and `st.secrets["HF_API_TOKEN"]` for the Gemini Vision and HuggingFace fallback stages. Neither is required for OCR or the color fallback to work, but both improve identification coverage.

1. Copy the template:
   ```bash
   mkdir -p .streamlit
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```
2. Get a free-tier **Gemini API key** from [Google AI Studio](https://aistudio.google.com/app/apikey) and paste it into `GEMINI_API_KEY`.
3. Get a free **HuggingFace access token** from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and paste it into `HF_API_TOKEN`.
4. `secrets.toml` is git-ignored — never commit real keys.

### 3. Run

```bash
streamlit run app.py
```

---

## 📁 Notebook vs. App

`Smart_Grocery_Recommender_CV.ipynb` is an early offline exploration built on the Instacart Market Basket Analysis dataset (3.4M orders, ~49,700 products, 206K users) with a stock, ImageNet-pretrained MobileNetV2 classifier. It is retained in the repository as an offline experimentation artifact but is architecturally distinct from, and superseded by, the deployed system: a curated 500-product catalog and the text-first, four-stage vision pipeline described above. See [`experiments/README.md`](experiments/README.md).

---

## 📄 Paper & Limitations

The full methodology, ablation study, K-sensitivity analysis, per-user-activity breakdown, and a worked recommendation example are documented in `PROJECT_REPORT.md`, adapted from the project's write-up. Known limitations (also discussed there): Precision@10 is bottlenecked by dataset scale rather than model quality; the HuggingFace and color-heuristic vision fallback stages are meaningfully weaker than OCR/Gemini; the hybrid weights (α, β) are fixed defaults rather than learned/grid-searched; and no formal labeled-image accuracy evaluation of the vision pipeline has been conducted.

---

## 👩‍💻 Author

**Dhwani Singhal**

[@dhwanisinghal-sudo](https://github.com/dhwanisinghal-sudo)
