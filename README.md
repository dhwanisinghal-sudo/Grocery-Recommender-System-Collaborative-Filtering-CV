# 🛒 Smart Grocery Recommender System

### Hybrid Collaborative Filtering + Multi-Stage Computer Vision

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Deployed-brightgreen?style=for-the-badge)

---

## 📌 Project Overview

A **Smart Grocery Recommendation System**, deployed as an interactive Streamlit app, that identifies grocery products from real-world photos and generates personalized recommendations from a hybrid collaborative-filtering engine.

**Users can:**

- 📷 Upload a photo of a grocery item and have it matched against a 500-product catalog through a 4-stage vision pipeline (OCR → Gemini Vision → Hugging Face → color heuristic)
- 🛒 Get **Top 10 personalized recommendations** via a hybrid of user-based CF, item-based CF, and SVD
- ❄️ Get popularity-based recommendations as a new user with no rating history
- 📊 Explore an evaluation-metrics dashboard, catalog/user search, and a raw-data explorer

Full methodology, architecture, and evaluation detail live in [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md) — this README is a quick-start summary.

---

## 🎯 Domain

**Machine Learning + Computer Vision**

**Application:** Smart Grocery Recommendation with Image Recognition, deployed as a Streamlit app (`app.py`)

---

## 📦 Dataset

| Detail               | Value                     |
| --------------------- | -------------------------- |
| Products               | 500                          |
| Categories                 | 13                              |
| Users                          | 150                                |
| Ratings                           | 6,796                                 |
| Rating scale                          | 1.5 – 5.0                                |
| Mean rating                               | 3.79                                        |
| Matrix sparsity                               | 90.9%                                          |
| Avg. ratings / user                               | 45.3                                                |

Catalog spans Personal Care, Dairy, Snacks, Spices, Drinks, Health, Home Care, Grains, Bakery, Frozen, Condiments, Beverages, and Noodles, including branded items (Amul, Parle, Britannia, MDH, Haldiram's, Patanjali, etc.).

Both `data/products_500plus.csv` and `data/user_ratings.csv` are curated/synthetic rather than scraped, and the generation process is documented and reproducible in code — see [`data/README.md`](data/README.md), `data/generate_catalog.py`, and `data/generate_ratings.py`.

---

## 🔁 Pipeline

```
📷 Image Upload
      ↓
🔎 4-Stage Vision Pipeline
   1. OCR (Tesseract) — match on-package text to keyword dictionary
   2. Gemini Vision fallback — constrained tag vocabulary + confidence
   3. Hugging Face Inference fallback — ImageNet-style labels, normalized
   4. Color-heuristic fallback — hue/brightness/texture rule-based guess
      ↓
🗂️ Catalog Match (500-product Indian grocery catalog)
      ↓
🤖 Hybrid Collaborative Filtering
   User-based CF + Item-based CF + SVD (rank-reciprocal blend)
      ↓
✅ Top 10 Personalized Recommendations
```

The vision pipeline has **not yet been formally accuracy-evaluated** end-to-end — no labeled image test set currently exists for this catalog (see `docs/PROJECT_REPORT.md` §8, Limitations).

---

## 🤖 Models

| Model                | Method                                                                 |
| --------------------- | ------------------------------------------------------------------------ |
| User-based CF          | Cosine similarity between user rating vectors, 15 nearest neighbors        |
| Item-based CF               | Cosine similarity between item vectors                                       |
| SVD                              | Truncated SVD, k=20 latent factors (`scipy.sparse.linalg.svds`)                  |
| **Hybrid (default)**                 | Rank-reciprocal blend: `α·CF_user + β·CF_item + (1−α−β)·SVD`, α=0.40, β=0.35 |

`α`/`β` defaults are grid-searched, not guesswork — `experiments/verify_grid_search.py` sweeps a 19-combo grid and the current defaults come out best by F1 on this dataset. Full detail in `docs/PROJECT_REPORT.md` §4.2.

---

## 📊 Evaluation Metrics

Computed via `compute_eval_metrics()` in `app.py`, on an 80/20 train-test split (seed=42), evaluated on a fixed *random* sample of 50 test users (`random.Random(42).sample(...)`).

| Metric               | SVD (zero-fill) | User-Based CF |
| ---------------------- | ------------------ | --------------- |
| RMSE                       | 3.61                  | 0.88            |

| Metric                | Value (K=10) |
| ------------------------ | -------------- |
| Precision@10                | 3.2%           |
| Recall@10                       | 4.5%           |
| F1                                    | 3.7%           |
| Catalog Coverage                          | 49.2%          |

Precision@10 stays low mainly because the dataset is small and 90.9% sparse — not because the underlying models are broken. See `docs/PROJECT_REPORT.md` §5 for the full discussion and `experiments/` for ablations (mean-centered SVD, K-sensitivity, grid search).

---

## ✅ Features

| Feature                | Description                                                   |
| ----------------------- | ----------------------------------------------------------------- |
| 🔍 Multi-stage vision    | OCR → Gemini Vision → Hugging Face → color heuristic, in priority order |
| 🤝 Hybrid CF            | User-based + Item-based CF + SVD, rank-reciprocal blend             |
| 👤 Personalized         | Recommendations based on user rating history                        |
| ❄️ Cold Start           | Popularity-based recommendations for new users                     |
| 📊 Evaluation Dashboard | RMSE, Precision/Recall/F1@K, Coverage, computed live in-app         |
| 🎛️ Interactive Widget   | User ID selector, image upload, adjustable α/β sliders              |
| 📐 Sparsity Analysis    | User-item matrix analysis                                           |
| 🔎 Search               | Catalog and user search, raw-data explorer                          |

---

## 🛠️ Tech Stack

| Category                | Libraries                                             |
| ------------------------ | -------------------------------------------------------- |
| Language                    | Python 3.x                                                    |
| Frontend                        | Streamlit                                                          |
| Collaborative Filtering             | scikit-learn, Pandas, NumPy, SciPy                                     |
| Computer Vision & OCR                   | OpenCV, Tesseract OCR, Pillow (PIL)                                        |
| AI Models                                   | Google Gemini API, Hugging Face Transformers                                  |
| Visualization                                   | Matplotlib, Plotly                                                                |
| Version Control                                     | Git, GitHub                                                                          |

---

## 📌 Key Results

| Metric                | Value                        |
| ------------------------ | ------------------------------- |
| Best CF setup               | Hybrid (User-CF + Item-CF + SVD)  |
| Best RMSE                       | 0.88 (User-Based CF)                 |
| Hybrid weights (α, β)               | 0.40, 0.35 — grid-search validated        |
| Catalog Coverage@10                     | 49.2%                                        |
| Cold Start                                  | ✅ Handled                                       |

---

## 🚀 Running the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

Reads `data/products_500plus.csv` and `data/user_ratings.csv` directly — no separate download or setup step needed.

---

## 📚 Further Reading

- [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md) — full architecture, models, evaluation, and known limitations
- [`data/README.md`](data/README.md) — dataset details and generation scripts
- [`experiments/`](experiments) — ablation, K-sensitivity, and grid-search verification scripts

---

## 🕰️ Earlier Exploratory Phase (Superseded)

> ⚠️ **Historical only.** The badges, numbers, and pipeline above describe the **current, deployed app** (`app.py`). Everything below describes a retired, architecturally unrelated notebook that predates and was superseded by that system — it does not reflect what's currently deployed.

The project's first exploratory phase used the public **Instacart Market Basket Analysis** dataset with a generic **MobileNetV2** ImageNet classifier, run as a Google Colab notebook, before moving to the curated 500-product catalog and purpose-built vision pipeline described above.

**Dataset (notebook phase):**

| Attribute      | Value                                                                                |
| --------------- | ---------------------------------------------------------------------------------------- |
| Source              | [Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis) |
| Total Orders            | 3,421,083                                                                                     |
| Total Products              | 49,688                                                                                            |
| Total Users                     | 206,209                                                                                               |
| Departments                         | 21                                                                                                        |
| Aisles                                  | 134                                                                                                           |

**Models (notebook phase):**

| Model                | RMSE       | Precision@10 | Recall@10  | F1 Score       |
| --------------------- | ---------- | ------------ | ---------- | -------------- |
| SVD                   | 1.7034     | 26.66%       | 18.50%     | 21.90%         |
| KNNBasic              | 2.1500     | 18.20%       | 12.30%     | 14.80%         |
| NMF                   | 1.9200     | 21.50%       | 15.60%     | 18.10%         |
| Hybrid (SVD+KNN)      | 1.6800     | 28.90%       | 20.10%     | 23.70%         |
| MobileNetV2 (ImageNet) | —         | —            | —          | 92.34% acc     |

**Why the project moved on:** this notebook validated the CV + CF concept at scale, but relied on a large third-party dataset not tied to a specific deployable catalog, and a generic ImageNet classifier not tuned to any particular product set or packaging. The project then moved to the curated catalog and multi-stage OCR/Gemini/Hugging Face vision pipeline described above, which is better suited to real product-package recognition and to deployment as an interactive app.

The notebook (`notebooks/Smart_Grocery_Recommender_(ML,CV).ipynb`) is retained in the repository for reference only and is not maintained or evaluated against current app data.

---

## 👩‍💻 Author

**Dhwani Singhal**

[@dhwanisinghal-sudo](https://github.com/dhwanisinghal-sudo)

---

<div align="center">

⭐ **If you like this project, don't forget to star the repository!**

</div>
