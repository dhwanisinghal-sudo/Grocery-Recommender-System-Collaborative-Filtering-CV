# notebooks/

This folder only holds my first attempt at the project. It is not the system
I deployed.

`Smart_Grocery_Recommender_(ML,CV)_ARCHIVED.ipynb` is a Google Colab notebook
with no UI. It used the Instacart Market Basket Analysis dataset, SVD via
`surprise`, and MobileNetV2 for image classification. I tried that approach,
looked at the results, and replaced it.

The deployed system is in the repo root:

| What | Where |
|---|---|
| The Streamlit app | `app.py` |
| Reference copy of the hyperparameters (`app.py` doesn't import it) | `config.py` |
| Data (500-product catalog, 150 users, 6,796 ratings) | `data/` |
| Architecture, evaluation and limitations | `docs/PROJECT_REPORT.md` |
| Scripts that reproduce the reported numbers | `experiments/` |
