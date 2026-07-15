# Smart Grocery Recommender

A Streamlit application that combines a multi-stage computer-vision product
identification pipeline with a hybrid collaborative-filtering recommendation
engine over a curated 500-product Indian grocery catalog.

> Full methodology and evaluation results: see
> `Smart_Grocery_Recommender_Paper_Corrected.docx` / the accompanying paper.

## What it does

1. **Identify a product from a photo.** Upload an image and the app resolves
   it to a catalog product through a four-stage fallback pipeline (see
   below), then uses that product as an anchor for "you might also like"
   suggestions.
2. **Recommend products for a user.** A hybrid collaborative-filtering
   engine blends three independent signals into a single ranked list.
3. **Explore and evaluate.** Search the catalog and rating data directly,
   and inspect the recommender's own accuracy metrics from a built-in
   evaluation dashboard.

## App modes

The app is a single-page Streamlit application (`app.py`, ~1,700 lines)
with seven sidebar modes:

- User recommendations (hybrid CF)
- Similar-product lookup (item-based CF)
- Image-based scanning
- Cold-start recommendations for new users
- Evaluation-metrics dashboard
- Catalog / user search
- Raw data explorer (Products, Ratings, Insights tabs)

## Vision pipeline

Product identification proceeds through four ordered stages, each attempted
only if the previous stage fails to produce a confident match:

1. **OCR (highest priority).** Tesseract extracts on-package text, matched
   against `GROCERY_KEYWORDS`, a curated dictionary of 150+ brand and
   product keywords mapped to product IDs.
2. **Gemini Vision fallback.** A structured prompt constrains output to a
   fixed vocabulary of allowed tags with confidence scores, plus a
   forbidden-tag list to suppress generic visual descriptors (e.g.
   "block", "foil", "rectangular").
3. **Hugging Face Inference fallback.** A general-purpose vision classifier
   supplies ImageNet-style labels, normalized onto the same controlled
   product-tag vocabulary.
4. **Color-heuristic fallback.** If both API stages are unavailable, a
   rule-based classifier inspects hue, brightness, and texture for a
   coarse guess.

A `DAIRY_SPECIFIC` priority-ordered, mutually-exclusive tag structure
disambiguates visually similar dairy products (butter, ghee, paneer, curd,
cheese, cream, milk). A manual text-search override lets a user correct a
misclassification directly.

## Recommendation engine

Three signals are computed independently and combined:

- **User-based CF** — cosine similarity between user rating vectors; the
  15 nearest neighbors' ratings are aggregated over items the target user
  hasn't rated.
- **Item-based CF** — cosine similarity between item vectors (columns of
  the rating matrix), restricted at inference time to `RELATED_CATEGORIES`
  for post-scan suggestions.
- **SVD matrix factorization** — truncated SVD (k=20 latent factors, via
  `scipy.sparse.linalg.svds`) reconstructed to a dense predicted-rating
  matrix.

The three ranked lists are combined via a rank-reciprocal hybrid score:

```
score(item) = α · (1 / rank_user_based)
            + β · (1 / rank_item_based)
            + (1 − α − β) · (1 / rank_svd)
```

with default weights `α = 0.40`, `β = 0.35`, both adjustable at runtime.
New users with no rating history receive popularity-based recommendations
(interaction count × average rating), filtered to their selected preferred
categories.

Data loading and all three CF models are wrapped in `@st.cache_data`, keyed
on a hash of the ratings table length, so models are computed once per data
version rather than recomputed on every interaction.

## Dataset

| Attribute            | Value        |
|-----------------------|--------------|
| Products              | 500          |
| Categories             | 13           |
| Users                  | 150          |
| Ratings                | 6,796        |
| Rating scale            | 1.5 – 5.0    |
| Mean rating            | 3.79         |
| Matrix sparsity         | 90.9%        |
| Avg. ratings / user      | 45.3         |

Catalog spans Personal Care, Dairy, Snacks, Spices, Drinks, Health, Home
Care, Grains, Bakery, Frozen, Condiments, Beverages, and Noodles, including
branded items (Amul, Parle, Britannia, MDH, Haldiram's, Patanjali, etc.).

Data files, expected under `data/`:
- `products_500plus.csv`
- `user_ratings.csv`

## 🛠️ Tech Stack

### Programming Language
- Python 3.x

### Frontend
- Streamlit

### Machine Learning & Recommendation
- Surprise (User-Based CF, Item-Based CF, SVD)
- Scikit-learn
- Pandas
- NumPy

### Computer Vision & OCR
- OpenCV
- Tesseract OCR
- Pillow (PIL)

### AI Models
- Google Gemini API
- Hugging Face Transformers

### Data Visualization
- Matplotlib
- Plotly

### Version Control
- Git
- GitHub

## Evaluation

The evaluation dashboard runs `compute_eval_metrics()` in `app.py` against
an 80/20 train-test split (seed=42) of the 6,796 ratings.

| Metric        | SVD (zero-fill) | User-Based CF |
|---------------|------------------|----------------|
| RMSE          | 3.61             | 0.88           |

| Metric            | Value (K=10) |
|--------------------|---------------|
| Precision@10       | 3.4%          |
| Recall@10          | 3.7%          |
| F1                 | 3.5%          |
| Catalog Coverage   | 48.2%         |

Precision@10 stays low mainly because the dataset is small and 90.9%
sparse — not because the underlying models are broken. Additional
ablations (mean-centered SVD, K-sensitivity, activity-level breakdown) are
reported in the paper but currently live in standalone analysis scripts,
not the deployed evaluation dashboard (see Limitations below).

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Vision fallback stages (Gemini, Hugging Face) require API credentials; the
app degrades gracefully to the color-heuristic stage if these are absent.

## Repository contents

- `app.py` — the deployed application described above.
- `data/products_500plus.csv`, `data/user_ratings.csv` — catalog and rating
  data used by the deployed app.
- `Smart_Grocery_Recommender_CV.ipynb` — an **earlier, superseded**
  offline experiment (Instacart dataset + MobileNetV2 classifier), retained
  for reference only. It is architecturally unrelated to the deployed
  vision pipeline in `app.py`. See `PROJECT_REPORT.md` for details on this
  earlier phase.

## Limitations / known gaps

- The vision pipeline has not been formally accuracy-evaluated (no labeled
  image test set currently exists for this catalog).
- Hybrid CF weights (α, β) are fixed defaults, not learned or
  grid-searched.
- Mean-centered SVD, K-sensitivity, and activity-level ablations exist as
  standalone scripts, not yet wired into the deployed evaluation dashboard.

## License / Attribution

See project paper for full references and methodology.
