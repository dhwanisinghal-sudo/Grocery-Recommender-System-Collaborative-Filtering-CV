"""
Verifies Section VIII-E: Sensitivity to Hybrid Weights (alpha, beta).

app.py's hybrid_recommend(alpha=0.4, beta=0.35, ...) defaults were never
grid-searched -- this script closes that gap.

It reconstructs the SAME three ranked lists the deployed app blends
(user-based CF, item-based CF, zero-fill SVD), built on the SAME 80/20
train/test split (seed=42) used everywhere else in experiments/, then
sweeps alpha x beta over a small grid and reports Precision@10/Recall@10/F1
for each combo on the same 50-user evaluation sample app.py currently uses
(the first 50 users in groupby insertion order -- see Gap #3; this is not
yet a random sample, so these numbers will shift slightly once that lands).

gamma (SVD weight) = 1 - alpha - beta, and the UI already enforces
alpha + beta <= 0.95, so gamma >= 0.05; combos violating that are skipped.

Run from the repo root or from experiments/:
    python experiments/verify_grid_search.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

THRESHOLD = 3.5
RANDOM_STATE = 42
SAMPLE_USERS = 50
TOP_N = 10
N_NEIGHBORS = 15

ALPHA_GRID = [0.20, 0.30, 0.40, 0.50, 0.60]
BETA_GRID = [0.15, 0.25, 0.35, 0.45]


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


def build_train_models():
    """Same split + same three base models app.py's hybrid_recommend blends,
    but fit on the TRAIN split only (app.py's live models are fit on all
    ratings; for evaluation we can't leak test ratings into the similarity
    matrices or the SVD factorization, so this mirrors verify_ablation.py /
    verify_k_sensitivity.py instead)."""
    ratings = pd.read_csv(find_data_file("user_ratings.csv"))
    train_r, test_r = train_test_split(ratings, test_size=0.2, random_state=RANDOM_STATE)

    train_pivot = train_r.pivot_table(index="user_id", columns="product_id", values="rating").fillna(0)
    train_mat = train_pivot.values.astype(float)

    user_sim_mat = cosine_similarity(csr_matrix(train_mat))
    item_sim_mat = cosine_similarity(csr_matrix(train_mat.T))
    user_sim_df = pd.DataFrame(user_sim_mat, index=train_pivot.index, columns=train_pivot.index)
    item_sim_df = pd.DataFrame(item_sim_mat, index=train_pivot.columns, columns=train_pivot.columns)

    k = min(20, min(train_mat.shape) - 1)
    U, sigma, Vt = svds(csr_matrix(train_mat), k=k)
    predicted = np.dot(np.dot(U, np.diag(sigma)), Vt)
    pred_df = pd.DataFrame(predicted, index=train_pivot.index, columns=train_pivot.columns)

    return train_pivot, user_sim_df, item_sim_df, pred_df, test_r


def user_based_recommend(uid, train_pivot, user_sim_df, top_n):
    if uid not in user_sim_df.index:
        return []
    sim_scores = user_sim_df[uid].drop(uid).sort_values(ascending=False).head(N_NEIGHBORS)
    rated = set(train_pivot.loc[uid][train_pivot.loc[uid] > 0].index)
    scores = {}
    for nb in sim_scores.index:
        w = sim_scores[nb]
        for pid, r in train_pivot.loc[nb].items():
            if r > 0 and pid not in rated:
                scores[pid] = scores.get(pid, 0) + w * r
    return sorted(scores, key=scores.get, reverse=True)[:top_n]


def item_based_recommend(uid, train_pivot, item_sim_df, top_n):
    if uid not in train_pivot.index:
        return []
    user_ratings = train_pivot.loc[uid]
    rated = user_ratings[user_ratings > 0]
    if rated.empty:
        return []
    scores = {}
    for pid, r in rated.items():
        if pid not in item_sim_df.index:
            continue
        for other, sim in item_sim_df[pid].drop(pid).items():
            if other not in rated.index:
                scores[other] = scores.get(other, 0) + sim * r
    return sorted(scores, key=scores.get, reverse=True)[:top_n]


def svd_recommend(uid, train_pivot, pred_df, top_n):
    if uid not in pred_df.index:
        return []
    rated = set(train_pivot.loc[uid][train_pivot.loc[uid] > 0].index)
    preds = pred_df.loc[uid].drop(list(rated), errors="ignore")
    return preds.sort_values(ascending=False).head(top_n).index.tolist()


def hybrid_recommend(uid, alpha, beta, train_pivot, user_sim_df, item_sim_df, pred_df, top_n=TOP_N):
    ub = user_based_recommend(uid, train_pivot, user_sim_df, top_n * 3)
    ib = item_based_recommend(uid, train_pivot, item_sim_df, top_n * 3)
    svd = svd_recommend(uid, train_pivot, pred_df, top_n * 3)
    gamma = 1 - alpha - beta
    scores = {}
    for rank, pid in enumerate(ub):
        scores[pid] = scores.get(pid, 0) + alpha * (1 / (rank + 1))
    for rank, pid in enumerate(ib):
        scores[pid] = scores.get(pid, 0) + beta * (1 / (rank + 1))
    for rank, pid in enumerate(svd):
        scores[pid] = scores.get(pid, 0) + gamma * (1 / (rank + 1))
    return sorted(scores, key=scores.get, reverse=True)[:top_n]


def evaluate_combo(alpha, beta, train_pivot, user_sim_df, item_sim_df, pred_df, test_r):
    test_grouped = (
        test_r[test_r["rating"] >= THRESHOLD]
        .groupby("user_id")["product_id"]
        .apply(set)
        .to_dict()
    )
    # NOTE: first 50, not a random sample -- matches app.py's current
    # (unfixed) methodology. See Gap #3.
    sample_uids = list(test_grouped.keys())[:SAMPLE_USERS]

    precisions, recalls = [], []
    for uid in sample_uids:
        if uid not in train_pivot.index:
            continue
        top_k = set(hybrid_recommend(uid, alpha, beta, train_pivot, user_sim_df, item_sim_df, pred_df))
        relevant = test_grouped.get(uid, set())
        hits = top_k & relevant
        precisions.append(len(hits) / TOP_N)
        recalls.append(len(hits) / len(relevant) if relevant else 0)

    p = float(np.mean(precisions)) if precisions else 0.0
    r = float(np.mean(recalls)) if recalls else 0.0
    f1 = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


def pct(x):
    return f"{x * 100:.2f}%"


def main():
    train_pivot, user_sim_df, item_sim_df, pred_df, test_r = build_train_models()

    results = []
    for alpha in ALPHA_GRID:
        for beta in BETA_GRID:
            if alpha + beta > 0.95:
                continue
            gamma = 1 - alpha - beta
            p, r, f1 = evaluate_combo(alpha, beta, train_pivot, user_sim_df, item_sim_df, pred_df, test_r)
            results.append({"alpha": alpha, "beta": beta, "gamma": gamma, "precision": p, "recall": r, "f1": f1})

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False).reset_index(drop=True)

    print(f"Table — Grid Search over Hybrid Weights (alpha x beta, gamma = 1 - alpha - beta)")
    print(f"Same 80/20 split (seed={RANDOM_STATE}), same {SAMPLE_USERS}-user sample, K={TOP_N}")
    print("-" * 70)
    print(f"{'alpha':<8}{'beta':<8}{'gamma':<8}{'Precision@10':<15}{'Recall@10':<15}{'F1':<10}")
    for _, row in results_df.iterrows():
        print(f"{row['alpha']:<8.2f}{row['beta']:<8.2f}{row['gamma']:<8.2f}"
              f"{pct(row['precision']):<15}{pct(row['recall']):<15}{pct(row['f1']):<10}")

    best = results_df.iloc[0]
    default_row = results_df[(results_df["alpha"] == 0.40) & (results_df["beta"] == 0.35)]

    print(f"\nBest by F1: alpha={best['alpha']:.2f}, beta={best['beta']:.2f}, "
          f"gamma={best['gamma']:.2f} -> F1={pct(best['f1'])}")
    if not default_row.empty:
        d = default_row.iloc[0]
        print(f"Current default (alpha=0.40, beta=0.35): "
              f"P={pct(d['precision'])}, R={pct(d['recall'])}, F1={pct(d['f1'])} "
              f"(rank {default_row.index[0] + 1} of {len(results_df)} by F1)")

    results_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grid_search_results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"\nFull results written to {results_csv}")


if __name__ == "__main__":
    main()
