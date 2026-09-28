"""
Closes the "50-user sample, no confidence interval" gap (Section VIII evaluation).

Instead of a single fixed 50-user sample, this evaluates EVERY user that has at
least one relevant (rating >= 3.5) item in the 20% test split, for all four
rankers (User-CF, Item-CF, zero-fill SVD, Hybrid alpha=0.40/beta=0.35), and
reports:
  1. Full-population Precision@10 / Recall@10 / F1 with a 95% bootstrap CI
     (users resampled with replacement, 5000 draws).
  2. The old 50-user-sample number (seed=42) next to it, so the difference
     between "sample" and "all users" is visible.
  3. Split-variance: Hybrid metrics over 10 different train/test split seeds
     (mean +/- std), since one 80/20 split is itself a single draw.

Same base models, same 80/20 split (seed=42), same threshold (3.5) and K=10 as
verify_grid_search.py, whose functions are imported directly.

Run from repo root:  python experiments/verify_full_user_ci.py
"""
import os, sys, random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_grid_search as g

ALPHA, BETA = 0.40, 0.35
K = g.TOP_N
N_BOOT = 5000


def build(seed):
    ratings = pd.read_csv(g.find_data_file("user_ratings.csv"))
    train_r, test_r = train_test_split(ratings, test_size=0.2, random_state=seed)
    tp = train_r.pivot_table(index="user_id", columns="product_id", values="rating").fillna(0)
    m = tp.values.astype(float)
    usim = pd.DataFrame(cosine_similarity(csr_matrix(m)), index=tp.index, columns=tp.index)
    isim = pd.DataFrame(cosine_similarity(csr_matrix(m.T)), index=tp.columns, columns=tp.columns)
    k = min(20, min(m.shape) - 1)
    U, s, Vt = svds(csr_matrix(m), k=k)
    pdf = pd.DataFrame(U @ np.diag(s) @ Vt, index=tp.index, columns=tp.columns)
    return tp, usim, isim, pdf, test_r


def per_user(tp, usim, isim, pdf, test_r, uids=None):
    rel = (test_r[test_r["rating"] >= g.THRESHOLD].groupby("user_id")["product_id"].apply(set).to_dict())
    uids = [u for u in (uids if uids is not None else rel) if u in pdf.index]
    out = {"UserCF": [], "ItemCF": [], "SVD": [], "Hybrid": []}
    for u in uids:
        relevant = rel.get(u, set())
        if not relevant:
            continue
        lists = {
            "UserCF": g.user_based_recommend(u, tp, usim, K),
            "ItemCF": g.item_based_recommend(u, tp, isim, K),
            "SVD": g.svd_recommend(u, tp, pdf, K),
            "Hybrid": g.hybrid_recommend(u, ALPHA, BETA, tp, usim, isim, pdf, K),
        }
        for name, recs in lists.items():
            hits = len(set(recs) & relevant)
            out[name].append((hits / K, hits / len(relevant)))
    return out


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) else 0.0


def summarize(arr):
    a = np.array(arr)
    return a[:, 0].mean(), a[:, 1].mean()


def bootstrap(arr, rng):
    a = np.array(arr)
    n = len(a)
    ps, rs, fs = [], [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        p, r = a[idx, 0].mean(), a[idx, 1].mean()
        ps.append(p); rs.append(r); fs.append(f1(p, r))
    q = lambda x: (np.percentile(x, 2.5), np.percentile(x, 97.5))
    return q(ps), q(rs), q(fs)


def main():
    tp, usim, isim, pdf, test_r = build(g.RANDOM_STATE)
    rel = test_r[test_r["rating"] >= g.THRESHOLD].groupby("user_id")["product_id"].apply(set).to_dict()
    sample = random.Random(g.RANDOM_STATE).sample(list(rel.keys()), min(g.SAMPLE_USERS, len(rel)))

    full = per_user(tp, usim, isim, pdf, test_r)
    samp = per_user(tp, usim, isim, pdf, test_r, uids=sample)
    n_full = len(full["Hybrid"]); n_samp = len(samp["Hybrid"])
    rng = np.random.default_rng(42)

    print(f"Users with >=1 relevant test item: {n_full} (of {tp.shape[0]} users in train pivot)")
    print(f"Old fixed sample: {n_samp} users\n")
    print("=" * 96)
    print(f"{'Model':<8} {'':<11} {'P@10':>8} {'R@10':>8} {'F1':>8}   {'95% CI P@10':<16} {'95% CI R@10':<16} {'95% CI F1':<16}")
    print("=" * 96)
    for name in ["UserCF", "ItemCF", "SVD", "Hybrid"]:
        ps, rs = summarize(samp[name]); fs_ = f1(ps, rs)
        print(f"{name:<8} {'50-sample':<11} {ps*100:7.2f}% {rs*100:7.2f}% {fs_*100:7.2f}%")
        p, r = summarize(full[name]); f_ = f1(p, r)
        (pl, ph), (rl, rh), (fl, fh) = bootstrap(full[name], rng)
        print(f"{'':<8} {'all '+str(n_full):<11} {p*100:7.2f}% {r*100:7.2f}% {f_*100:7.2f}%   "
              f"[{pl*100:5.2f},{ph*100:5.2f}]  [{rl*100:5.2f},{rh*100:5.2f}]  [{fl*100:5.2f},{fh*100:5.2f}]")
    print("-" * 96)

    # Paired comparison Hybrid vs each single model (bootstrap on per-user P@10 difference)
    print("\nPaired bootstrap: Hybrid minus baseline, per-user Precision@10 difference (95% CI):")
    h = np.array(full["Hybrid"])[:, 0]
    for name in ["UserCF", "ItemCF", "SVD"]:
        b = np.array(full[name])[:, 0]
        d = h - b
        boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        verdict = "CI excludes 0" if (lo > 0 or hi < 0) else "CI includes 0 -> not distinguishable"
        print(f"  Hybrid - {name:<7}: {d.mean()*100:+.2f} pp  [{lo*100:+.2f}, {hi*100:+.2f}]  {verdict}")

    # Split variance
    print("\nSplit variance: Hybrid over 10 different 80/20 split seeds (all evaluable users each):")
    rows = []
    for seed in range(1, 11):
        m = build(seed)
        res = per_user(*m)
        p, r = summarize(res["Hybrid"])
        rows.append((p, r, f1(p, r)))
    a = np.array(rows)
    for i, (p, r, f_) in enumerate(rows, 1):
        print(f"  seed {i:>2}: P@10={p*100:5.2f}%  R@10={r*100:5.2f}%  F1={f_*100:5.2f}%")
    print(f"  mean +/- std: P@10={a[:,0].mean()*100:.2f}+/-{a[:,0].std(ddof=1)*100:.2f}  "
          f"R@10={a[:,1].mean()*100:.2f}+/-{a[:,1].std(ddof=1)*100:.2f}  "
          f"F1={a[:,2].mean()*100:.2f}+/-{a[:,2].std(ddof=1)*100:.2f}")


if __name__ == "__main__":
    main()
