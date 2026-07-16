"""
Verifies Section VIII-B: Ablation — Mean-Centered SVD.

Reproduces Table IV of the paper by running the SAME train/test split,
K, and evaluation logic as app.py's compute_eval_metrics() (zero-fill
baseline), then re-running SVD after mean-centering each user's ratings
before factorization (adding the mean back, clipped to [1.5, 5.0]).

Run from the repo root or from experiments/:
    python experiments/verify_ablation.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

RATING_MIN, RATING_MAX = 1.5, 5.0
K = 10
THRESHOLD = 3.5
RANDOM_STATE = 42
SAMPLE_USERS = 50


def find_data_file(filename):
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        filename,
        os.path.join("data", filename),
        os.path.join(here, "..", "data", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find {filename}. Searched: {candidates}")


def load_split():
    ratings = pd.read_csv(find_data_file("user_ratings.csv"))
    train_r, test_r = train_test_split(ratings, test_size=0.2, random_state=RANDOM_STATE)
    train_pivot = train_r.pivot_table(index="user_id", columns="product_id", values="rating").fillna(0)
    test_pivot = test_r.pivot_table(index="user_id", columns="product_id", values="rating").fillna(0)
    return train_r, test_r, train_pivot, test_pivot


def factorize(train_mat):
    k = min(20, min(train_mat.shape) - 1)
    U, sigma, Vt = svds(csr_matrix(train_mat), k=k)
    return np.dot(np.dot(U, np.diag(sigma)), Vt)


def evaluate(pred_df, train_pivot, test_pivot, test_r):
    # RMSE over co-observed (user, product) test cells
    cu = [u for u in test_pivot.index if u in pred_df.index]
    cp = [p for p in test_pivot.columns if p in pred_df.columns]
    actual_flat = test_pivot.loc[cu, cp].values.flatten()
    pred_flat = pred_df.loc[cu, cp].values.flatten()
    mask = actual_flat > 0
    rmse = float(np.sqrt(mean_squared_error(actual_flat[mask], pred_flat[mask])))

    # Precision/Recall/F1@K and coverage, first 50 test users with rating >= threshold
    test_grouped = (
        test_r[test_r["rating"] >= THRESHOLD]
        .groupby("user_id")["product_id"]
        .apply(set)
        .to_dict()
    )
    precisions, recalls = [], []
    for uid in list(test_grouped.keys())[:SAMPLE_USERS]:
        if uid not in pred_df.index:
            continue
        rated_train = set(train_pivot.loc[uid][train_pivot.loc[uid] > 0].index)
        preds_uid = pred_df.loc[uid].drop(list(rated_train), errors="ignore")
        top_k = set(preds_uid.sort_values(ascending=False).head(K).index)
        relevant = test_grouped.get(uid, set())
        hits = top_k & relevant
        precisions.append(len(hits) / K)
        recalls.append(len(hits) / len(relevant) if relevant else 0)

    p_at_k = float(np.mean(precisions)) if precisions else 0.0
    r_at_k = float(np.mean(recalls)) if recalls else 0.0
    f1 = float(2 * p_at_k * r_at_k / (p_at_k + r_at_k)) if (p_at_k + r_at_k) > 0 else 0.0

    all_recommended = set()
    for uid in list(pred_df.index)[:SAMPLE_USERS]:
        rated = set(train_pivot.loc[uid][train_pivot.loc[uid] > 0].index)
        top = pred_df.loc[uid].drop(list(rated), errors="ignore").sort_values(ascending=False).head(K).index
        all_recommended.update(top)
    coverage = float(len(all_recommended) / len(train_pivot.columns))

    return {"rmse": rmse, "precision_at_k": p_at_k, "recall_at_k": r_at_k, "f1": f1, "coverage": coverage}


def run_zero_fill(train_mat, train_pivot, test_pivot, test_r):
    predicted = factorize(train_mat)
    pred_df = pd.DataFrame(predicted, index=train_pivot.index, columns=train_pivot.columns)
    return evaluate(pred_df, train_pivot, test_pivot, test_r)


def run_mean_centered(train_mat, train_pivot, test_pivot, test_r):
    # Mean over each user's RATED items only (zeros are "unrated", not "rated zero")
    mat = train_mat.copy()
    rated_mask = mat > 0
    row_sums = (mat * rated_mask).sum(axis=1)
    row_counts = rated_mask.sum(axis=1)
    row_means = np.divide(row_sums, row_counts, out=np.zeros_like(row_sums), where=row_counts > 0)

    centered = mat.copy()
    centered[rated_mask] = mat[rated_mask] - np.repeat(row_means, row_counts)

    predicted = factorize(centered)
    predicted = predicted + row_means[:, None]  # add each user's mean back
    predicted = np.clip(predicted, RATING_MIN, RATING_MAX)

    pred_df = pd.DataFrame(predicted, index=train_pivot.index, columns=train_pivot.columns)
    return evaluate(pred_df, train_pivot, test_pivot, test_r)


def pct(x):
    return f"{x * 100:.2f}%"


def main():
    train_r, test_r, train_pivot, test_pivot = load_split()
    train_mat = train_pivot.values.astype(float)

    zero_fill = run_zero_fill(train_mat, train_pivot, test_pivot, test_r)
    mean_centered = run_mean_centered(train_mat, train_pivot, test_pivot, test_r)

    print("Table IV — Ablation: Zero-Fill vs. Mean-Centered SVD")
    print("-" * 60)
    print(f"{'Metric':<18}{'Zero-Fill SVD':<18}{'Mean-Centered SVD':<18}")
    print(f"{'RMSE':<18}{zero_fill['rmse']:<18.4f}{mean_centered['rmse']:<18.4f}")
    print(f"{'Precision@10':<18}{pct(zero_fill['precision_at_k']):<18}{pct(mean_centered['precision_at_k']):<18}")
    print(f"{'Recall@10':<18}{pct(zero_fill['recall_at_k']):<18}{pct(mean_centered['recall_at_k']):<18}")
    print(f"{'F1 Score':<18}{pct(zero_fill['f1']):<18}{pct(mean_centered['f1']):<18}")
    print(f"{'Coverage':<18}{pct(zero_fill['coverage']):<18}{pct(mean_centered['coverage']):<18}")

    print("\nPaper (Table IV) reference values, for comparison:")
    print(f"{'Metric':<18}{'Zero-Fill SVD':<18}{'Mean-Centered SVD':<18}")
    print(f"{'RMSE':<18}{'3.6145':<18}{'0.8789':<18}")
    print(f"{'Precision@10':<18}{'3.40%':<18}{'3.40%':<18}")
    print(f"{'Recall@10':<18}{'3.66%':<18}{'5.51%':<18}")
    print(f"{'F1 Score':<18}{'3.53%':<18}{'4.21%':<18}")
    print(f"{'Coverage':<18}{'48.2%':<18}{'40.6%':<18}")


if __name__ == "__main__":
    main()
