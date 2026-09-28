"""
Re-runs the alpha/beta grid from verify_grid_search.py on ALL evaluable users
(not the fixed 50-user sample) and reports where the default (0.40, 0.35)
ranks, plus how far apart the top combos really are.

Same split (seed=42), threshold (3.5), K=10 and the same three base rankers as
verify_grid_search.py. The three ranked lists are computed once per user and
re-blended for each (alpha, beta), so this is fast.

Run from repo root:  python experiments/verify_grid_full_users.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_grid_search as g

K = g.TOP_N


def blend(ub, ib, svd, a, b):
    s = {}
    for r, p in enumerate(ub):  s[p] = s.get(p, 0) + a * (1 / (r + 1))
    for r, p in enumerate(ib):  s[p] = s.get(p, 0) + b * (1 / (r + 1))
    for r, p in enumerate(svd): s[p] = s.get(p, 0) + (1 - a - b) * (1 / (r + 1))
    return sorted(s, key=s.get, reverse=True)[:K]


def main():
    tp, usim, isim, pdf, test_r = g.build_train_models()
    rel = test_r[test_r["rating"] >= g.THRESHOLD].groupby("user_id")["product_id"].apply(set).to_dict()
    uids = [u for u in rel if u in pdf.index and rel[u]]
    lists = {u: (g.user_based_recommend(u, tp, usim, K * 3),
                 g.item_based_recommend(u, tp, isim, K * 3),
                 g.svd_recommend(u, tp, pdf, K * 3)) for u in uids}

    rows = []
    for a in g.ALPHA_GRID:
        for b in g.BETA_GRID:
            if 1 - a - b < 0.05 - 1e-9:
                continue
            ps, rs = [], []
            for u in uids:
                hits = len(set(blend(*lists[u], a, b)) & rel[u])
                ps.append(hits / K); rs.append(hits / len(rel[u]))
            p, r = np.mean(ps), np.mean(rs)
            f = 2 * p * r / (p + r) if p + r else 0
            rows.append((a, b, 1 - a - b, p, r, f))
    rows.sort(key=lambda x: -x[5])

    print(f"All {len(uids)} evaluable users, {len(rows)} valid (alpha, beta) combos\n")
    print(f"{'rank':<5}{'alpha':>6}{'beta':>6}{'gamma':>7}{'P@10':>8}{'R@10':>8}{'F1':>8}")
    for i, (a, b, gm, p, r, f) in enumerate(rows, 1):
        mark = "  <- default" if (abs(a - .4) < 1e-9 and abs(b - .35) < 1e-9) else ""
        print(f"{i:<5}{a:6.2f}{b:6.2f}{gm:7.2f}{p*100:7.2f}%{r*100:7.2f}%{f*100:7.2f}%{mark}")
    fs = [x[5] for x in rows]
    print(f"\nF1 range across the grid: {min(fs)*100:.2f}% - {max(fs)*100:.2f}%  (spread {100*(max(fs)-min(fs)):.2f} pp)")


if __name__ == "__main__":
    main()
