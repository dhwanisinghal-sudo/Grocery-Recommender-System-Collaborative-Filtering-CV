# 📋 Smart Grocery Recommender System: Project Report

**Domain:** Machine Learning + Computer Vision
**Application:** Image-based product identification followed by hybrid collaborative-filtering recommendations
**Status:** Deployed as a Streamlit app

> This report covers the current system (`app.py`). A paper draft accompanies
> the project but is **not included in this repository**; the scripts in
> `experiments/` reproduce its tables (see
> [`experiments/VERIFICATION_README.md`](../experiments/VERIFICATION_README.md)),
> and [`README.md`](../README.md) has a short summary for users. My first attempt, with the Instacart dataset
> and a MobileNetV2 classifier, is described in **§9** for reference only. I
> replaced it.

---

## 1. Problem Statement

Most grocery apps show everyone the same popular products. They ignore what a
person has bought before, and they don't help when you're holding a packet and
want to know what it is. I built a system that:

- identifies a grocery product from a photo using a four-stage vision pipeline,
- recommends products using a hybrid of user-based CF, item-based CF and SVD,
- ties the two together as scan → identify → recommend, in a Streamlit app.

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

The catalog covers Personal Care, Dairy, Snacks, Spices, Drinks, Health, Home
Care, Grains, Bakery, Frozen, Condiments, Beverages and Noodles, and includes
branded items (Amul, Parle, Britannia, MDH, Haldiram's, Patanjali and others).
I curated the product names and categories by hand. The ratings are synthetic
(see `data/README.md` and `data/generate_ratings.py`).

Data files are in `data/`:

- `products_500plus.csv`
- `user_ratings.csv`

---

## 3. System Architecture

```
📷 Image Upload
      ↓
🔎 4-Stage Vision Pipeline
   1. OCR (Tesseract): match on-package text to GROCERY_KEYWORDS
   2. Gemini Vision fallback: constrained tag vocabulary + confidence
   3. Hugging Face Inference fallback: ImageNet-style labels, normalized
   4. Color-heuristic fallback: hue/brightness/texture rule-based guess
      ↓
🗂️ Catalog Match (500-product Indian grocery catalog)
      ↓
🤖 Hybrid Collaborative Filtering
   User-based CF + Item-based CF + SVD (rank-reciprocal blend)
      ↓
✅ Personalized Recommendations
```

Some dairy products look almost the same in a photo, so I added a
`DAIRY_SPECIFIC` structure: a priority-ordered set of mutually exclusive tags
for butter, ghee, paneer, curd, cheese, cream and milk. There is also a manual
text-search override, so a user can correct a wrong classification directly.

---

## 4. Models Used

### 4.1 Vision Pipeline

| Stage                 | Method                                                             | Role                                             |
| ---------------------- | -------------------------------------------------------------------- | -------------------------------------------------- |
| 1: OCR                    | Tesseract, matched against a 200+ entry keyword dictionary                | Tried first                                            |
| 2: Gemini Vision              | Google Gemini API, constrained tag + confidence output                       | Fallback if OCR finds nothing                              |
| 3: Hugging Face                    | General-purpose vision classifier, normalized onto the product-tag vocabulary        | Fallback if Gemini is unavailable or finds nothing                  |
| 4: Color heuristic                       | Rule-based hue/brightness/texture classifier                                         | Last resort if both API stages are unavailable                     |

I evaluated the pipeline on a labeled test set of 260 images
(`data/vision_test_set/`, 20 images for each of the 13 categories), using
`experiments/verify_vision_accuracy.py`. The labels are at category level.
There are two runs. In the first I had no `GEMINI_API_KEY` (an `HF_API_TOKEN`
was set), so OCR, Hugging Face and the color fallback all ran. The second run
had both keys and is the current reference result:

| Stage | Images resolved | Accuracy |
| ------- | ------------------ | ---------- |
| OCR Text Detection | 130/260 (50%) | 84.6% |
| Gemini Vision | 128/260 (49%) | 92.2% |
| Hugging Face Vision | 2/260 (1%) | 0.0% |
| **Overall** | 260/260 | **87.7%** |

Overall accuracy went from 51.5% (first run, no Gemini key) to 87.7% once
Gemini was available. Gemini resolved nearly half the images, the ones OCR
couldn't, at 92.2%. Hugging Face was only reached for 2 images, which is too
few to say anything about it. The color fallback never triggered: OCR, Gemini
and HF between them resolved all 260 images. Per-category accuracy runs from
70% (Condiments, Health, Spices) to 100% (Beverages, Dairy). The per-image
results are in `experiments/vision_accuracy_results.csv`.

For comparison, this is the first run (no Gemini key). It's the only run where
Hugging Face and the color fallback resolved a meaningful number of images:

| Stage | Images resolved | Accuracy |
| ------- | ------------------ | ---------- |
| OCR Text Detection | 128/260 (49%) | 84.4% |
| Hugging Face Vision | 99/260 (38%) | 24.2% |
| Color Fallback | 33/260 (13%) | 6.1% |
| **Overall** | 260/260 | **51.5%** |

Its per-image results are in git history (commit `84131e0`).

The two runs don't let me judge the fallback stages fairly. In the reference
run Hugging Face resolved only 2 images and the color heuristic none. In the
first run their 24.2% and 6.1% are measured only on the images OCR could not
resolve, which are harder than average. I have not run Hugging Face or the
color heuristic on their own over all 260 images, so their standalone accuracy
is unmeasured.

**Error analysis of the reference run (32 wrong of 260).** 21 of the 32 errors
are predicted as *Dairy*. Re-running the OCR stage locally on those 21 images
shows why:

- 15 were resolved at the OCR stage, and all 15 have a dairy word among the
  extracted tags. `detect_dairy_type()` (in `app.py`) gives the matching dairy
  products a +200 score boost, and that outweighs any other tag. Three of them
  are salt images: the check `tag in kt` treats the tag `salt` as a match for
  the dairy keyword `salted butter`, because `salt` is a substring of `salted`,
  so Sea Salt and Salt-pouch images land on Amul Butter. Others are
  ingredient-word cases: peanut butter → butter, a health drink that
  mentions milk, Kurkure (cheese flavour), a paneer masala, paneer momos.
- The other 6 were resolved by Gemini and predicted Dairy directly.

I have **not** fixed this in `app.py`, because a fix changes the numbers above
and they would need a fresh run with API keys. Tightening the substring test
to a word-boundary match, and not applying the dairy boost when a more
specific non-dairy tag is present, is the obvious next step.

Both runs measure category-level accuracy: whether the matched product is in
the right one of 13 categories. They don't measure per-SKU accuracy against
the exact 500-product catalog, because I don't have per-product ground truth.
That is future work (see §8).

### 4.2 Collaborative Filtering Models

| Model                     | Method                                                                                                                |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| User-based CF                  | Cosine similarity between user rating vectors; the 15 nearest neighbors are aggregated over unrated items                            |
| Item-based CF                       | Cosine similarity between item vectors; restricted to `RELATED_CATEGORIES` for post-scan suggestions                             |
| SVD                                       | Truncated SVD with k=20 latent factors (`scipy.sparse.linalg.svds`), reconstructed into a dense predicted-rating matrix                  |

The three ranked lists are combined with a rank-reciprocal hybrid score:

```
score(item) = α · (1 / rank_user_based)
            + β · (1 / rank_item_based)
            + (1 − α − β) · (1 / rank_svd)
```

The default weights are `α = 0.40` and `β = 0.35`. They were set by hand: they
first appear in the git history on 19 June (commit `331b2e0`), three months
before any tuning. The grid search (`experiments/verify_grid_search.py`, added
25 September, results in `experiments/grid_search_results.csv`) came afterwards
as a check. Its grid is 5 × 4 in steps of 0.10 (19 valid combinations with
γ ≥ 0.05), not 0.05 steps. The defaults rank first by F1 on the original
50-user sample, and still rank first on all 149 evaluable users
(`experiments/verify_grid_full_users.py`). But their margin over the runner-up
is 0.02 percentage points, and the whole grid spans only 0.91 pp of F1
(3.45%–4.36%), much less than the width of the confidence intervals in §5. The
data therefore do not single out a best weighting: the defaults are reasonable,
not shown to be optimal. They stay fixed by default and can be changed in the
app. New users
with no rating history get popularity-based recommendations (interaction count
× average rating), filtered to their preferred categories.

---

## 5. Evaluation Metrics

These come from `compute_eval_metrics()` in `app.py`, on an 80/20 train-test
split (seed 42) of the 6,796 ratings. Precision@10, Recall@10, F1 and Coverage
are computed over a fixed random sample of 50 test users
(`random.Random(42)`). Earlier I took the first 50 users in `groupby()` order,
which wasn't a random sample, and I fixed that.

| Metric | SVD (zero-fill) | User-Based CF |
| ------- | ----------------- | --------------- |
| RMSE      | 3.61                 | 0.88            |

| Metric                | Value (K=10) |
| ------------------------ | -------------- |
| Precision@10                | 3.2%           |
| Recall@10                       | 4.5%           |
| F1                                    | 3.7%           |
| Catalog Coverage                          | 49.2%          |

### 5.1 Full-population evaluation with confidence intervals

The 50-user sample above is a point estimate. `experiments/verify_full_user_ci.py`
evaluates all 149 users who have at least one relevant (rating ≥ 3.5) item in
the test split, with the same split, threshold and K, for all four rankers.
Confidence intervals are 95% bootstrap intervals over users (5,000 draws).

| Model | 50-user F1 | All 149 users: P@10 | R@10 | F1 [95% CI] |
| ------- | ------------ | --------------------- | ------ | -------------- |
| User-based CF | 3.85% | 2.68% | 4.79% | 3.44% [2.26, 4.65] |
| Item-based CF | 3.50% | 3.36% | 4.64% | 3.89% [2.85, 5.02] |
| SVD (zero-fill) | 3.73% | 3.09% | 4.25% | 3.57% [2.49, 4.77] |
| **Hybrid (0.40 / 0.35)** | 5.81% | 3.56% | 5.64% | 4.36% [3.09, 5.68] |

- The hybrid's 5.81% F1 on the 50-user sample was optimistic. On all 149 users
  it is 4.36%.
- Paired bootstrap on per-user Precision@10, hybrid minus baseline: vs User-CF
  +0.87 pp [+0.27, +1.54] (excludes 0); vs Item-CF +0.20 pp [−0.54, +0.94];
  vs SVD +0.47 pp [−0.34, +1.34]. So the hybrid beats User-CF, but it is not
  distinguishable from Item-CF or SVD on this data.
- One 80/20 split is itself a single draw. Over 10 different split seeds the
  hybrid gets P@10 3.51 ± 0.44%, R@10 5.25 ± 0.45%, F1 4.20 ± 0.46% (mean ± std).

Precision@10 is low. My explanation is that the dataset is small and 90.9%
sparse, but I haven't tested that directly. Mean-centered SVD, K-sensitivity
and the α/β grid search are in `experiments/` as scripts, and also in the
app's Evaluation Metrics mode, which has four tabs: Baseline, Ablation,
K-Sensitivity and α/β Grid Search.

---

## 6. App Modes

The app is a single-page Streamlit application (`app.py`, around 2,000 lines)
with seven sidebar modes:

- User recommendations (hybrid CF)
- Similar-product lookup (item-based CF)
- Image-based scanning
- Cold-start recommendations for new users
- Evaluation-metrics dashboard
- Catalog / user search
- Raw data explorer (Products, Ratings and Insights tabs)

---

## 7. Tech Stack

| Category                     | Libraries                                                              |
| ------------------------------ | -------------------------------------------------------------------------- |
| Language                          | Python 3.x                                                                     |
| Frontend                              | Streamlit                                                                          |
| Collaborative Filtering                   | scikit-learn (cosine similarity), SciPy (`svds`), Pandas, NumPy               |
| Vision & OCR                         | Tesseract via `pytesseract`, Pillow                                                         |
| External APIs                                         | Google Gemini API, Hugging Face Inference API (both called with `requests`)                                                    |
| Version Control                                           | Git, GitHub                                                                                             |

---

## 8. Results Summary & Limitations

**Results:**

- The vision pipeline tries up to four stages to turn a product photo into a
  catalog item. It prefers the more precise stages (OCR, then the
  constrained-tag Gemini stage) over the last-resort color heuristic.
- User-based CF beats zero-fill SVD on this dataset (RMSE 0.88 vs 3.61). The
  ablation (`experiments/verify_ablation.py`) shows that most of that gap comes
  from zero-filling missing ratings, not from neighborhood methods being better
  in themselves: mean-centered SVD also reaches an RMSE of 0.88.
- Data loading and all three CF models are cached (`@st.cache_data`, keyed on
  the length of the ratings table), so the models are computed once per data
  version and not on every interaction.

**Known gaps:**

- All ratings are synthetic (`data/README.md`, `data/generate_ratings.py`). The
  metrics therefore show how the pipeline behaves on this dataset and how the
  variants compare with each other. They don't show real-world recommendation
  accuracy. I haven't collected any real-user ratings.
- The α, β defaults were set before any tuning. The later grid search (§4.2)
  covers 19 combinations in steps of 0.10. The defaults rank first, but the
  grid is nearly flat (0.91 pp of F1 across all 19), so this is not evidence
  that they are optimal.
- The hybrid is not shown to beat Item-based CF or SVD (§5.1). Only its
  advantage over User-based CF is statistically clear.
- 21 of the 32 vision errors are predicted as Dairy, mostly from
  `detect_dairy_type()` (§4.1). Documented, not fixed.
- The activity-level ablation (Precision/Recall/F1 for heavy vs. light raters)
  is only described. I haven't built it as a script or a dashboard tab (see
  `experiments/VERIFICATION_README.md`).
- The 50-user metrics in the app dashboard are point estimates. Confidence
  intervals for the full population are in §5.1 (`verify_full_user_ci.py`),
  but the vision accuracy in §4.1 has no interval.

**Future work (not done in this project):**

- **Per-SKU vision accuracy.** The 87.7% in §4.1 is category-level: whether the
  matched product is in the right one of 13 categories. Measuring per-SKU
  accuracy against the exact 500-product catalog would need photos of the
  actual individual products (not representative category photos), each
  labeled with its exact `product_id`. That is a much bigger data-collection
  job than the current 260-image category set, so I left it as future work.

---

## 9. Earlier Exploratory Phase (Superseded)

> ⚠️ **Historical only.** This section describes a retired notebook that has
> nothing to do with the deployed app in §1–8. It came first, and I replaced
> it with the system described above.

My first version used the public **Instacart Market Basket Analysis** dataset
and a generic **MobileNetV2** ImageNet classifier, run as a Google Colab
notebook. After that I moved to a curated catalog I could deploy, and to a
vision pipeline built for grocery packaging.

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

**Why I moved on:** the notebook showed that the CV + CF idea works at scale,
but it depended on a large third-party dataset that wasn't tied to a catalog I
could deploy, and on a generic ImageNet classifier that wasn't tuned to any
particular products or packaging. So I built the 500-product catalog and the
OCR/Gemini/Hugging Face pipeline in §1–8, which suits real product packaging
and works as an interactive app.

The notebook is still in the repo (under `notebooks/`) for reference. I don't
maintain it or evaluate it against the current app data.
