# Project Report — Earlier Phase (Superseded)

> **This document describes an earlier, exploratory phase of the project
> and does not reflect the current deployed system.** The deployed
> application is `app.py`, a Streamlit app over a curated 500-product
> catalog with a text-first vision pipeline and a hybrid
> user-based/item-based/SVD recommender — see `README.md` and the project
> paper (`Smart_Grocery_Recommender_Paper_Corrected.docx`) for the current
> system and its evaluation results.
>
> This earlier phase is retained in the repository only as the offline
> notebook `Smart_Grocery_Recommender_CV.ipynb`, for reference. It was not
> integrated into the deployed system and none of its numbers describe
> `app.py`.

## What this phase explored

An initial exploration of grocery product recognition and recommendation
using the public Instacart Market Basket Analysis dataset together with a
stock, ImageNet-pretrained MobileNetV2 image classifier.

### Dataset (this phase only)

| Attribute   | Value                          |
|-------------|----------------------------------|
| Source      | Instacart Market Basket Analysis (Kaggle) |
| Orders      | ~3.4M                            |
| Products    | ~49,700                          |
| Users       | ~206,000                          |

### Vision approach (this phase only)

A stock MobileNetV2 image classifier (ImageNet-pretrained), reported at
92.34% accuracy on its own evaluation setup. This is a single-model
appearance-only classifier, architecturally distinct from the deployed
system's four-stage, text-first vision pipeline (OCR → Gemini →
Hugging Face → color heuristic).

### Collaborative filtering approach (this phase only)

Exploratory work with SVD, KNN, and NMF over the Instacart interaction
data, reported RMSE ~1.70 (hybrid variant, ~1.68). These figures are **not
comparable** to the deployed system's results (RMSE 0.88, user-based CF)
since they come from a different dataset, different sparsity profile, and
different modeling code entirely.

### Interface (this phase only)

A Jupyter notebook (`Smart_Grocery_Recommender_CV.ipynb`), used for offline
experimentation — not a deployed or interactive application.

## Why this phase was set aside

The Instacart dataset, while large, does not match the curated Indian
grocery catalog and branded-product identification problem the project
ultimately targeted. The MobileNetV2 classifier also struggled on
fine-grained, packaging-heavy product categories without domain-specific
fine-tuning, which motivated the shift to a text-first (OCR-priority)
vision pipeline in the deployed system.

This exploration is preserved as-is in the notebook for reference and as a
possible direction for future work at larger data scale, but it is not
part of, and does not describe, the current deployed application.

## Current system

For the actual deployed system — architecture, vision pipeline, hybrid CF
formula, dataset, and evaluation results — see:

- `README.md` (this repository)
- `Smart_Grocery_Recommender_Paper_Corrected.docx` (full paper with
  methodology, evaluation, ablations, and limitations)
