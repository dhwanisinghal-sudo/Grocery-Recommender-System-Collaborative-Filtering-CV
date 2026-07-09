# 📋 Smart Grocery Recommender System — Project Report

**Domain:** Recommender Systems + Computer Vision
**Dataset:** Curated Indian grocery catalog (500 products, 150 users, 6,796 ratings)
**Status:** ✅ Complete — deployed as a Streamlit app

> Adapted from the project write-up *"Smart Grocery Recommender: A Hybrid Collaborative Filtering System with a Multi-Stage Vision Pipeline for Personalized Grocery Recommendations"* (Dhwani Singhal, Independent Project). This document mirrors that paper's structure and figures for repo-level reference.

---

## 1. Problem Statement

Online grocery platforms face a discovery problem on two fronts: they recommend generically-popular products regardless of individual preference, and they offer no way to identify a product from a photograph when a user doesn't know its exact catalog name or brand spelling. This is a particular friction point in markets with a high density of regional and private-label brands, where a single visual category (e.g. "biscuit") maps to dozens of near-identical packages.

This project addresses both problems in a single Streamlit application that (1) identifies a grocery item from an uploaded photo through a four-stage fallback pipeline, and (2) recommends personalized items using a hybrid collaborative-filtering engine trained on real user-item ratings. The two concerns are deliberately kept separate: image identification maps a photo to a catalog product, and collaborative filtering maps a user's history to a ranked recommendation list, with the identified product serving as a similarity anchor for "you might also like" suggestions.

**Contributions:**
1. A practical, layered vision pipeline that prioritizes package text over visual classification, motivated by the empirical unreliability of appearance-only classifiers on packaged goods.
2. A hybrid CF engine combining three complementary signal sources (user-based CF, item-based CF, SVD).
3. A transparent evaluation, including an ablation study that isolates and corrects a specific failure mode of naive SVD on sparse rating matrices.

---

## 2. Dataset

| Attribute | Value |
|---|---|
| Products | 500 |
| Categories | 13 — Personal Care, Dairy, Snacks, Spices, Drinks, Health, Home Care, Grains, Bakery, Frozen, Condiments, Beverages, Noodles |
| Users | 150 |
| Ratings | 6,796 explicit ratings |
| Rating scale | 1.5 – 5.0 |
| Mean rating | 3.79 |
| Matrix sparsity | 90.9% |
| Avg. ratings / user | 45.3 |
| Avg. ratings / product | 13.6 |
| Key files | `data/products_500plus.csv`, `data/user_ratings.csv` |

An earlier phase of this project explored the public Instacart Market Basket Analysis dataset (3.4M orders, ~49,700 products, 206K users) together with a stock, ImageNet-pretrained MobileNetV2 classifier. That exploration is retained in the repository as an offline experimentation notebook (see §9) but is architecturally distinct from, and superseded by, the deployed system described here, which uses the curated catalog above and a different, text-first vision pipeline (§4).

---

## 3. System Architecture

The system is a single-page Streamlit application (~1,400 lines) with **seven interactive modes**: user recommendations, similar-product lookup, image-based scanning, cold-start recommendations for new users, an evaluation-metrics dashboard, catalog/user search, and a raw data explorer.

An uploaded image is passed through the vision pipeline to produce a ranked set of candidate tags; tags are matched against a hand-built keyword-to-product-ID index; the resulting anchor products drive an item-based similarity lookup restricted to related categories, producing "you might also like" suggestions, while independently a hybrid collaborative-filtering pass produces the primary personalized recommendation list.

All models (user-similarity matrix, item-similarity matrix, SVD decomposition) are computed in-memory at runtime from the two source CSVs using Streamlit's `@st.cache_data`, keyed on a hash of the ratings-table length — computed once per data version and reused across a session, rather than pre-trained and serialized. This keeps the system simple to redeploy and keeps evaluation fully reproducible directly from the shared data files.

---

## 4. Design Rationale: OCR-First Vision Pipeline

Rather than relying on a single image classifier, product identification proceeds through **four ordered stages**, each attempted only if the previous stage fails to produce a confident match:

1. **OCR (highest priority).** Tesseract extracts on-package text at multiple page-segmentation modes; extracted text is matched against a curated dictionary of 150+ brand and product keywords (e.g. "Amul," "Maggi," "MDH") mapped to catalog product IDs. Packaging text is largely invariant to lighting and camera angle, so this is treated as the most reliable signal when available.
2. **Gemini Vision fallback.** If OCR yields no match, the image is sent to a Gemini vision-language model with a structured prompt constraining output to a fixed vocabulary of allowed product tags with confidence scores, plus an explicit forbidden-tag list to suppress generic visual descriptors ("block," "foil," "rectangular," etc.).
3. **HuggingFace Inference fallback.** A general-purpose vision classifier (tried across several hosted models) supplies raw ImageNet-style labels, mapped through a label-normalization dictionary onto the same controlled product-tag vocabulary.
4. **Color-heuristic fallback.** If all API-based stages fail (no network, missing credentials), a rule-based classifier inspects average hue, brightness, and pixel-value texture to make a coarse guess (e.g. bright yellow-green with high texture variance suggests noodles or chips).

A manual text-search override is also available in the UI, letting a user correct a misclassification directly. This ordering trades a small amount of latency (OCR is cheap and local) for meaningfully better reliability on the packaged-goods use case this app targets: lightweight CNNs and general vision classifiers are known to struggle on fine-grained, packaging-heavy product categories without domain-specific fine-tuning, which is exactly the gap OCR-first identification is designed to close.

---

## 5. Collaborative Filtering Engine

Three recommendation signals are computed independently and combined:

- **User-based CF:** cosine similarity between all user rating vectors over the full item space; for a target user, the 15 nearest neighbors' ratings are aggregated as a similarity-weighted sum over items the target hasn't rated.
- **Item-based CF:** cosine similarity between item vectors (columns of the rating matrix); for each item a user has rated, similar unrated items accumulate a similarity-weighted score.
- **SVD matrix factorization:** the rating matrix is decomposed via truncated SVD (k=20 latent factors, `scipy.sparse.linalg.svds`) and reconstructed into a dense predicted-rating matrix.

These three ranked lists are combined via a **rank-reciprocal hybrid**: each item's hybrid score is

```
α · (1 / rank in user-based list) + β · (1 / rank in item-based list) + (1 − α − β) · (1 / rank in SVD list)
```

with default weights **α = 0.40** and **β = 0.35**, both adjustable at runtime via sidebar sliders. New users with no rating history receive popularity-based recommendations (interaction count × average rating), filtered to their selected preferred categories, as a standard cold-start fallback.

---

## 6. Cross-Category Recommendations: `RELATED_CATEGORIES`

After an image is matched to a product, the "you might also like" panel doesn't just recommend more items from the *same* category — it recommends from a curated set of **related** categories (`RELATED_CATEGORIES` in `app.py`), preventing semantically unrelated cross-category noise. For example, a detected snack pulls suggestions from Snacks, Bakery, and Frozen, but not Home Care. The keyword-matching layer (`GROCERY_KEYWORDS`) additionally uses auxiliary structures — `JUNK_DESCRIPTORS`, `CUISINE_NOISE` — to suppress generic visual descriptors, and a priority-ordered, mutually-exclusive tag structure (`DAIRY_SPECIFIC`) to disambiguate visually similar dairy products (butter, ghee, paneer, curd, cheese, cream, milk) so that, for example, a "curd" tag doesn't also trigger a butter match.

Session state (`st.session_state`) persists an image scan's results (tags, matched product IDs, method, debug log) across reruns, so switching sidebar controls or using the manual-search override doesn't require re-running the vision pipeline.

---

## 7. Evaluation Setup

The CF engine is evaluated on an 80/20 random train-test split (seed=42) of the 6,796 ratings. RMSE is computed on the intersection of test users and products present in the training-fitted prediction matrix. Ranking quality is assessed via Precision@10 and Recall@10 over a fixed sample of 50 test users, using a relevance threshold of rating ≥ 3.5; catalog coverage is the fraction of the 500-product catalog appearing in any user's top-10 list within that same sample. All numbers below were computed directly from the app's own evaluation code against the real rating data — not estimated or simulated.

---

## 8. Results

### 8.A Baseline Evaluation

SVD, factorized directly on the zero-filled rating matrix — the code path currently in `compute_eval_metrics()` — produces a markedly worse RMSE than user-based CF:

| Metric | SVD (zero-fill) | User-Based CF |
|---|---|---|
| RMSE | 3.6145 | 0.8820 |

Ranking and coverage metrics at K=10 (SVD-based recommendations, 50-user sample):

| Metric | Value |
|---|---|
| Precision@10 | 3.40% |
| Recall@10 | 3.66% |
| F1 Score | 3.53% |
| Catalog Coverage | 48.2% |

### 8.B Ablation: Mean-Centered SVD

**Hypothesis:** SVD's inflated RMSE is an artifact of zero-filling unrated entries — treating "unrated" as "rated zero" biases the factorization toward under-predicting every cell. **Test:** re-run SVD after subtracting each user's mean rating (over their rated items only) prior to factorization, then adding it back to the reconstructed predictions, clipped to [1.5, 5.0].

| Metric | Zero-Fill SVD | Mean-Centered SVD |
|---|---|---|
| RMSE | 3.6145 | **0.8789** |
| Precision@10 | 3.40% | 3.40% |
| Recall@10 | 3.66% | **5.51%** |
| F1 Score | 3.53% | **4.21%** |
| Coverage | 48.2% | 40.6% |

Mean-centering fully closes the RMSE gap with user-based CF (3.61 → 0.88, matching 0.88) and improves Recall@10 by 49% and F1 by 19% relative to the zero-fill baseline — confirming the original RMSE gap was a scale artifact, not a genuine model-quality difference. Precision@10 is unchanged; coverage drops modestly (48.2% → 40.6%), indicating mean-centering sharpens rather than broadens the set of items SVD confidently ranks highly.

**Note:** this mean-centered variant is a reported ablation/analysis, not (yet) the code path executing in the deployed app's `compute_eval_metrics()`, which currently reports the zero-fill numbers in §8.A. Integrating mean-centering into the deployed evaluation is a natural next step (§10).

### 8.C Sensitivity to K

Precision, Recall, and F1 at K ∈ {5, 10, 20} on the same 50-user sample (zero-fill SVD, consistent with §8.A):

| K | Precision | Recall | F1 |
|---|---|---|---|
| 5 | 4.00% | 2.30% | 2.92% |
| 10 | 3.40% | 3.66% | 3.53% |
| 20 | 4.10% | 11.82% | 6.09% |

Recall rises sharply from K=10 to K=20 (3.66% → 11.82%) while precision stays roughly flat, indicating many relevant items sit just outside the top-10 boundary — a wider window captures them without diluting hit rate. With only 500 products, a K=20 list already covers 4% of the entire catalog for a single user, so the marginal cost of widening K is low relative to the recall gained.

### 8.D Breakdown by User Activity Level

Users were split into two groups by median rating count (45 ratings/user):

| Group | Precision@10 | Recall@10 |
|---|---|---|
| Heavy raters (≥ median, n=77) | 3.77% | 4.14% |
| Light raters (< median, n=73) | 2.36% | 4.36% |

Heavy raters see higher precision (3.77% vs. 2.36%) and a modestly higher F1 (3.95% vs. 3.06%), consistent with more training signal per user improving ranking quality. Light raters show slightly higher recall (4.36% vs. 4.14%), likely because they have fewer held-out relevant items in the test split, making each hit worth proportionally more. This gap is smaller than the near-3x difference in typical rating counts between groups would suggest, indicating the dominant bottleneck is total dataset scale (150 users overall) rather than per-user history length specifically.

### 8.E Worked Example

User **U097** (59 ratings, above the median) has top-rated items including Del Monte Spaghetti (Noodles, 5.0), Boost Health Drink (Health, 5.0), and Real Fruit Juice Mixed Fruit (Drinks, 5.0) — a profile weighted toward Health and Drinks. Top-5 recommendations from each individual model, alongside the final hybrid output (α=0.40, β=0.35):

| Rank | User-Based CF | Item-Based CF | SVD | Hybrid (final) |
|---|---|---|---|---|
| 1 | Red Label Tea | Moong Dal | Toned Milk | **Toned Milk** |
| 2 | Dabur Honey | Toned Milk | Aamras Juice | Red Label Tea |
| 3 | Amul Lassee | Bikaji Bhujia | Choco Milk | Moong Dal |
| 4 | Cheese Spread | Aamras Juice | Urad Dal | Aamras Juice |
| 5 | Greek Yogurt | Urad Dal | Guava Juice | Dabur Honey |

Mother Dairy Toned Milk appears in both Item-Based CF's and SVD's individual top-5 lists (though not User-Based CF's), and the rank-reciprocal hybrid correctly surfaces it as the top overall recommendation — agreement between two of three signals dominating the combined ranking. None of the three individual lists exactly matches the final hybrid ranking, showing the hybrid genuinely integrates rank information across all three models rather than deferring to its highest-weighted component (User-Based CF, α=0.40).

---

## 9. Relationship to `Smart_Grocery_Recommender_CV.ipynb`

This repo also contains an earlier notebook, `Smart_Grocery_Recommender_CV.ipynb`, built during initial exploration using the Instacart Market Basket Analysis dataset (3.4M orders, ~49,700 products, 206K users) with a stock MobileNetV2 classifier and scikit-surprise (SVD/KNN/NMF) for CF. It is **not** part of the deployed system described above, uses an entirely different dataset and CV approach, and was not integrated into the deployed system. It is retained for reference in `experiments/` — see the note there — and should not be read as documenting the current app's methodology or results.

---

## 10. Discussion, Limitations, and Future Work

The central empirical result — a ~4x RMSE inflation attributable entirely to a zero-fill artifact rather than genuine model weakness — has a practical implication beyond this system: RMSE comparisons between rating-prediction models are only meaningful if all models handle missing entries consistently. User-Based CF here naturally skips missing entries (predictions only made when neighbor overlap exists), while naive SVD implicitly penalizes them, so an unadjusted head-to-head RMSE comparison systematically disadvantages factorization-based methods on sparse data — a useful methodological note for small-scale recommender deployments, where sparsity is often far higher than in large commercial datasets.

The K-sensitivity (§8.C) and activity-level breakdown (§8.D) both point to the same underlying cause for persistently low Precision@10: the dataset is simply too small, in both users and interactions, to give any top-10 ranking a large enough pool of confidently-relevant candidates. This is a data-scale problem, not an algorithmic one — the worked example (§8.E) shows the models behaving sensibly and producing category-coherent, mutually-corroborating recommendations even when the aggregate precision metric looks weak.

**Known limitations and next steps:**
- Precision@10 remains low (3.4%) even after correcting the SVD scaling issue, primarily due to dataset scale (150 users, 90.9% sparsity) and a 3.5 threshold that captures the majority of ratings (mean 3.79), enlarging the relevant-item set relative to any fixed K. Larger user data, implicit feedback signals (views, cart-adds) alongside explicit ratings, and regularized factorization (e.g. ALS with L2 regularization) are natural next steps.
- The vision pipeline's later fallback stages (HuggingFace, color-heuristic) are meaningfully weaker than OCR and Gemini; a fine-tuned, catalog-specific image classifier would likely improve identification when package text is occluded or absent (e.g. loose produce).
- The hybrid CF weights (α, β) are currently fixed defaults rather than learned or grid-searched; a systematic, ideally cross-validated hyperparameter search is left for future work.
- No formal accuracy evaluation of the vision pipeline has been conducted, since it would require a labeled image test set that doesn't currently exist for this catalog — flagged explicitly as a gap rather than presenting an informal estimate as a validated result.
- The offline Instacart/MobileNetV2 exploration (§9) was not integrated into the deployed system and represents a separate direction, potentially combinable with the current architecture at much larger data scale, rather than a component of the results reported here.

---

## 11. Conclusion

This project combines a text-first, multi-stage vision pipeline with a hybrid collaborative-filtering engine, evaluated transparently on real interaction data. The central empirical finding is methodological: naive SVD on a sparse, zero-filled rating matrix produces a misleadingly poor RMSE, and mean-centering fully resolves this discrepancy while also improving ranking recall — a useful, generalizable caution for small-scale recommender-system deployments. Concrete paths toward closing the remaining precision gap include larger data, implicit feedback, regularized factorization, and a fine-tuned vision model.

---

## References

[1] B. Sarwar, G. Karypis, J. Konstan, and J. Riedl, "Item-based collaborative filtering recommendation algorithms," in Proc. 10th Int. Conf. World Wide Web (WWW), 2001, pp. 285–295.

[2] Y. Koren, R. Bell, and C. Volinsky, "Matrix factorization techniques for recommender systems," IEEE Computer, vol. 42, no. 8, pp. 30–37, 2009.

[3] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: Inverted residuals and linear bottlenecks," in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2018, pp. 4510–4520.

[4] Instacart, "Instacart Market Basket Analysis," Kaggle, 2017. Available: https://www.kaggle.com/c/instacart-market-basket-analysis

[5] Y. Koren, "Factor in the neighbors: Scalable and accurate collaborative filtering," ACM Trans. Knowl. Discov. Data, vol. 4, no. 1, pp. 1–24, 2010.

[6] X. He, L. Liao, H. Zhang, L. Nie, X. Hu, and T.-S. Chua, "Neural collaborative filtering," in Proc. 26th Int. Conf. World Wide Web (WWW), 2017, pp. 173–182.

[7] R. Smith, "An overview of the Tesseract OCR engine," in Proc. 9th Int. Conf. Document Analysis and Recognition (ICDAR), 2007, pp. 629–633.

[8] A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," in Proc. Int. Conf. Learning Representations (ICLR), 2021.
