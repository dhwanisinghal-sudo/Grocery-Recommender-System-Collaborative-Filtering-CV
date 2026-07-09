# 📦 Dataset Setup

This project uses a **curated, synthetic Indian grocery catalog** — not the Instacart Market Basket Analysis dataset referenced in the earlier exploratory notebook (`experiments/Smart_Grocery_Recommender_CV.ipynb`).

Both data files are committed to this repo and require no download:

| File | Description |
|---|---|
| `products_500plus.csv` | 500 products across 13 categories, with brand/name/price/rating/tags |
| `user_ratings.csv` | 6,796 explicit ratings from 150 users on a 1.5–5.0 scale |

## 📊 Dataset Stats

| Metric | Value |
|---|---|
| Products | 500 |
| Categories | 13 |
| Users | 150 |
| Ratings | 6,796 |
| Rating scale | 1.5 – 5.0 |
| Mean rating | 3.79 |
| Matrix sparsity | 90.9% |
| Avg. ratings / user | 45.3 |
| Avg. ratings / product | 13.6 |

## Loading

`app.py` reads both files directly from this folder at runtime (via `find_file()`), so no separate download or setup step is needed — just run:

```bash
streamlit run app.py
```

from the repo root, with these two CSVs present in `data/`.

---

> The Instacart-based setup previously documented here (orders.csv, order_products__prior.csv, products.csv, ~1.3GB total) belonged to the earlier offline exploration notebook only, and does not apply to the deployed app. See `experiments/README.md`.
