"""
Verifies Section VIII-E: Worked Example (user U097).

Reproduces Table VI of the paper: top-5 recommendations from User-Based CF,
Item-Based CF, SVD, and the final rank-reciprocal hybrid (alpha=0.40,
beta=0.35, gamma=0.25), for user U097 — computed with the SAME models
(built on the full ratings matrix, not a train/test split) and the SAME
recommender functions as the deployed app (app.py: user_based_recommend,
item_based_recommend, svd_recommend, hybrid_recommend).

Run from the repo root or from experiments/:
    python experiments/verify_worked_example.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

USER_ID = "U097"
TOP_N = 5
ALPHA = 0.40  # User-Based CF weight
BETA = 0.35   # Item-Based CF weight
# gamma (SVD weight) = 1 - ALPHA - BETA = 0.25, derived


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


def build_models():
    products = pd.read_csv(find_data_file("products_500plus.csv"))
    ratings = pd.read_csv(find_data_file("user_ratings.csv"))

    pivot = ratings.pivot_table(index="user_id", columns="product_id", values="rating").fillna(0)
    mat = pivot.values.astype(float)

    user_sim_mat = cosine_similarity(csr_matrix(mat))
    item_sim_mat = cosine_similarity(csr_matrix(mat.T))
    user_sim_df = pd.DataFrame(user_sim_mat, index=pivot.index, columns=pivot.index)
    item_sim_df = pd.DataFrame(item_sim_mat, index=pivot.columns, columns=pivot.columns)

    k = min(20, min(mat.shape) - 1)
    U, sigma, Vt = svds(csr_matrix(mat), k=k)
    predicted = np.dot(np.dot(U, np.diag(sigma)), Vt)
    pred_df = pd.DataFrame(predicted, index=pivot.index, columns=pivot.columns)

    product_map = products.set_index("product_id").to_dict("index")
    return pivot, user_sim_df, item_sim_df, pred_df, product_map


def user_based_recommend(pivot, user_sim_df, user_id, top_n=10, n_neighbors=15):
    if user_id not in user_sim_df.index:
        return []
    sim_scores = user_sim_df[user_id].drop(user_id).sort_values(ascending=False).head(n_neighbors)
    rated = set(pivot.loc[user_id][pivot.loc[user_id] > 0].index)
    scores = {}
    for nb in sim_scores.index:
        w = sim_scores[nb]
        for pid, r in pivot.loc[nb].items():
            if r > 0 and pid not in rated:
                scores[pid] = scores.get(pid, 0) + w * r
    return sorted(scores, key=scores.get, reverse=True)[:top_n]


def item_based_recommend(pivot, item_sim_df, user_id, top_n=10):
    if user_id not in pivot.index:
        return []
    user_ratings = pivot.loc[user_id]
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


def svd_recommend(pivot, pred_df, user_id, top_n=10):
    if user_id not in pred_df.index:
        return []
    rated = set(pivot.loc[user_id][pivot.loc[user_id] > 0].index)
    preds = pred_df.loc[user_id].drop(list(rated), errors="ignore")
    return preds.sort_values(ascending=False).head(top_n).index.tolist()


def hybrid_recommend(ub, ib, svd, top_n=10, alpha=ALPHA, beta=BETA):
    scores = {}
    for rank, pid in enumerate(ub):
        scores[pid] = scores.get(pid, 0) + alpha * (1 / (rank + 1))
    for rank, pid in enumerate(ib):
        scores[pid] = scores.get(pid, 0) + beta * (1 / (rank + 1))
    gamma = 1 - alpha - beta
    for rank, pid in enumerate(svd):
        scores[pid] = scores.get(pid, 0) + gamma * (1 / (rank + 1))
    return sorted(scores, key=scores.get, reverse=True)[:top_n]


def names(product_map, pids):
    return [product_map.get(pid, {}).get("name", pid) for pid in pids]


def main():
    pivot, user_sim_df, item_sim_df, pred_df, product_map = build_models()

    n_ratings = int((pivot.loc[USER_ID] > 0).sum())
    print(f"User {USER_ID}: {n_ratings} ratings")

    ub = user_based_recommend(pivot, user_sim_df, USER_ID, top_n=TOP_N * 3)
    ib = item_based_recommend(pivot, item_sim_df, USER_ID, top_n=TOP_N * 3)
    svd = svd_recommend(pivot, pred_df, USER_ID, top_n=TOP_N * 3)
    hybrid = hybrid_recommend(ub, ib, svd, top_n=TOP_N)

    ub_top5, ib_top5, svd_top5 = ub[:TOP_N], ib[:TOP_N], svd[:TOP_N]

    print("\nTable VI — Worked Example: Top-5 Recommendations for U097")
    print("-" * 80)
    print(f"{'Rank':<6}{'User-Based CF':<20}{'Item-Based CF':<20}{'SVD':<20}{'Hybrid (final)':<20}")
    for i in range(TOP_N):
        row = [
            str(i + 1),
            names(product_map, [ub_top5[i]])[0] if i < len(ub_top5) else "-",
            names(product_map, [ib_top5[i]])[0] if i < len(ib_top5) else "-",
            names(product_map, [svd_top5[i]])[0] if i < len(svd_top5) else "-",
            names(product_map, [hybrid[i]])[0] if i < len(hybrid) else "-",
        ]
        print(f"{row[0]:<6}{row[1]:<20}{row[2]:<20}{row[3]:<20}{row[4]:<20}")

    print("\nPaper (Table VI) reference values, for comparison:")
    ref = [
        ("Red Label Tea", "Moong Dal", "Toned Milk", "Toned Milk"),
        ("Dabur Honey", "Toned Milk", "Aamras Juice", "Red Label Tea"),
        ("Amul Lassee", "Bikaji Bhujia", "Choco Milk", "Moong Dal"),
        ("Cheese Spread", "Aamras Juice", "Urad Dal", "Aamras Juice"),
        ("Greek Yogurt", "Urad Dal", "Guava Juice", "Dabur Honey"),
    ]
    print(f"{'Rank':<6}{'User-Based CF':<20}{'Item-Based CF':<20}{'SVD':<20}{'Hybrid (final)':<20}")
    for i, row in enumerate(ref):
        print(f"{i+1:<6}{row[0]:<20}{row[1]:<20}{row[2]:<20}{row[3]:<20}")


if __name__ == "__main__":
    main()
