"""
Verifies Section VIII-C: Sensitivity to K.

Reproduces Table V of the paper: Precision/Recall/F1 at K in {5, 10, 20},
using the same zero-fill SVD predictions as the baseline (Section VIII-A),
on the same 50-user evaluation sample.

Run from the repo root or from experiments/:
    python experiments/verify_k_sensitivity.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

THRESHOLD = 3.5
RANDOM_STATE = 42
SAMPLE_USERS = 50
K_VALUES = [5, 10, 20]


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


def build_predictions():
    ratings = pd.read_csv(find_data_file("user_ratings.csv"))
    train_r, test_r = train_test_split(ratings, test_size=0.2, random_state=RANDOM_STATE)
    train_pivot = train_r.pivot_table(index="user_id", columns="product_id", values="rating").fillna(0)
    train_mat = train_pivot.values.astype(float)

    k = min(20, min(train_mat.shape) - 1)
    U, sigma, Vt = svds(csr_matrix(train_mat), k=k)
    predicted = np.dot(np.dot(U, np.diag(sigma)), Vt)
    pred_df = pd.DataFrame(predicted, index=train_pivot.index, columns=train_pivot.columns)

    return train_pivot, pred_df, test_r


def precision_recall_f1_at_k(train_pivot, pred_df, test_r, K):
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

    p = float(np.mean(precisions)) if precisions else 0.0
    r = float(np.mean(recalls)) if recalls else 0.0
    f1 = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


def pct(x):
    return f"{x * 100:.2f}%"


def main():
    train_pivot, pred_df, test_r = build_predictions()

    print("Table V — Sensitivity to K (zero-fill SVD, same 50-user sample)")
    print("-" * 55)
    print(f"{'K':<6}{'Precision':<14}{'Recall':<14}{'F1':<14}")
    for K in K_VALUES:
        p, r, f1 = precision_recall_f1_at_k(train_pivot, pred_df, test_r, K)
        print(f"{K:<6}{pct(p):<14}{pct(r):<14}{pct(f1):<14}")

    print("\nPaper (Table V) reference values, for comparison:")
    print(f"{'K':<6}{'Precision':<14}{'Recall':<14}{'F1':<14}")
    print(f"{'5':<6}{'4.00%':<14}{'2.30%':<14}{'2.92%':<14}")
    print(f"{'10':<6}{'3.40%':<14}{'3.66%':<14}{'3.53%':<14}")
    print(f"{'20':<6}{'4.10%':<14}{'11.82%':<14}{'6.09%':<14}")


if __name__ == "__main__":
    main()
