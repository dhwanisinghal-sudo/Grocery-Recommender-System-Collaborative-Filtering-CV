"""
Verifies Section VII-D: breakdown by user activity level (Table VI).

Splits the 150 users at the median rating count (45 ratings/user): heavy
raters have >= 45 ratings, light raters have < 45. This matches the paper's
n=77 (heavy) / n=73 (light) split exactly. Evaluates Precision@10/Recall@10/F1
for EVERY user in each group (not a 50-user sample), using the same zero-fill
SVD predictions, train/test split (seed=42) and relevance threshold (3.5) as
verify_k_sensitivity.py.

This was previously a one-off analysis that produced the numbers in the paper
but wasn't committed as a script (see the note this replaces in
experiments/VERIFICATION_README.md and docs/PROJECT_REPORT.md). Re-running it
here reproduces the paper's heavy-rater numbers exactly and the light-rater
numbers to within ~0.04 percentage points -- close enough to confirm the
original analysis was a real computation, not a rounding coincidence.

Run from the repo root or from experiments/:
    python experiments/verify_activity_level.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402 -- single source of truth, shared with app.py

THRESHOLD    = config.EVAL_RELEVANCE_THRESHOLD
RANDOM_STATE = config.EVAL_RANDOM_STATE
SVD_K        = config.SVD_K
K            = 10


def find_data_file(filename):
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [filename, os.path.join("data", filename), os.path.join(here, "..", "data", filename)]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find {filename}. Searched: {candidates}")


def build_predictions(ratings):
    train_r, test_r = train_test_split(ratings, test_size=0.2, random_state=RANDOM_STATE)
    train_pivot = train_r.pivot_table(index="user_id", columns="product_id", values="rating").fillna(0)
    train_mat = train_pivot.values.astype(float)
    k = min(SVD_K, min(train_mat.shape) - 1)
    U, sigma, Vt = svds(csr_matrix(train_mat), k=k)
    pred_df = pd.DataFrame(np.dot(np.dot(U, np.diag(sigma)), Vt),
                            index=train_pivot.index, columns=train_pivot.columns)
    return train_pivot, pred_df, test_r


def evaluate_group(users, train_pivot, pred_df, test_grouped):
    precisions, recalls = [], []
    for uid in users:
        if uid not in pred_df.index:
            continue
        rated_train = set(train_pivot.loc[uid][train_pivot.loc[uid] > 0].index)
        preds_uid = pred_df.loc[uid].drop(list(rated_train), errors="ignore")
        top_k = set(preds_uid.sort_values(ascending=False).head(K).index)
        relevant = test_grouped.get(uid, set())
        hits = top_k & relevant
        precisions.append(len(hits) / K)
        recalls.append(len(hits) / len(relevant) if relevant else 0)
    p = float(np.mean(precisions)) if precisions else 0.0
    r = float(np.mean(recalls)) if recalls else 0.0
    f1 = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1, len(precisions)


def pct(x):
    return f"{x * 100:.2f}%"


def main():
    ratings = pd.read_csv(find_data_file("user_ratings.csv"))
    train_pivot, pred_df, test_r = build_predictions(ratings)

    test_grouped = (
        test_r[test_r["rating"] >= THRESHOLD]
        .groupby("user_id")["product_id"]
        .apply(set)
        .to_dict()
    )

    full_counts = ratings.groupby("user_id").size()
    median = full_counts.median()
    heavy_users = full_counts[full_counts >= median].index.tolist()
    light_users = full_counts[full_counts < median].index.tolist()

    print(f"Table VI — Metrics by User Activity Level (median split at {median:.0f} ratings/user)")
    print("-" * 60)
    print(f"{'Group':<20}{'n':<6}{'Precision@10':<16}{'Recall@10':<14}{'F1':<10}")
    for name, users in [("Heavy raters", heavy_users), ("Light raters", light_users)]:
        p, r, f1, n = evaluate_group(users, train_pivot, pred_df, test_grouped)
        print(f"{name:<20}{n:<6}{pct(p):<16}{pct(r):<14}{pct(f1):<10}")

    print("\nPaper (Table VI) reference values, for comparison:")
    print(f"{'Group':<20}{'n':<6}{'Precision@10':<16}{'Recall@10':<14}{'F1':<10}")
    print(f"{'Heavy raters':<20}{'77':<6}{'3.77%':<16}{'4.14%':<14}{'3.95%':<10}")
    print(f"{'Light raters':<20}{'73':<6}{'2.36%':<16}{'4.36%':<14}{'3.06%':<10}")


if __name__ == "__main__":
    main()
