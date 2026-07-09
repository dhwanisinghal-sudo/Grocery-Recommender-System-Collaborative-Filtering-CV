# Experiments — Offline Exploration (Instacart)

`Smart_Grocery_Recommender_CV.ipynb` is an early offline exploration phase of this project, built on the public **Instacart Market Basket Analysis** dataset (3.4M orders, ~49,700 products, 206K users) using a stock, ImageNet-pretrained **MobileNetV2** classifier for image recognition and **scikit-surprise** (SVD/KNN/NMF) for collaborative filtering.

**This notebook is superseded by, and architecturally unrelated to, the deployed Streamlit app** (`app.py` at the repo root). The deployed app uses:
- a different, curated 500-product synthetic Indian grocery catalog (not Instacart), and
- a different, text-first, four-stage vision pipeline (OCR → Gemini Vision → HuggingFace → color heuristic), not MobileNetV2.

It's kept here for reference on the project's evolution, not as documentation of the current system's methodology or results. See the repo root `README.md` and `PROJECT_REPORT.md` for the deployed system.
