# notebooks/ — archived material only

Everything in this folder is historical. It does **not** represent the
currently deployed system.

- `archived_phase1_instacart_mobilenet/` — the project's first exploration
  phase: Instacart Market Basket Analysis dataset + SVD (via `surprise`) +
  MobileNetV2 image classification, run in Google Colab with no UI. This
  approach was evaluated and then replaced.

## Where the real thing is

The deployed system is the Streamlit app in the repo root:

| Component | Location |
|---|---|
| Live application | `app.py` |
| Model hyperparameters (single source of truth) | `config.py` |
| Data (500-product catalog, 150-user/6,796-rating set) | `data/` |
| Architecture, evaluation methodology, known limitations | `docs/PROJECT_REPORT.md` |
| Reproducible verification scripts for reported metrics | `experiments/` |

If you're reviewing this project, start with `docs/PROJECT_REPORT.md`, not
the notebook.
