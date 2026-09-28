"""
Demonstrates what changing alpha and beta does to the hybrid recommender.

For one user (default U097, the same user as verify_worked_example.py) it
computes the three base ranked lists ONCE (User-CF, Item-CF, zero-fill SVD,
built on the full ratings matrix, like the deployed app), then re-blends them
with the formula

    score(item) = alpha * 1/(rank_user + 1)
                + beta  * 1/(rank_item + 1)
                + (1 - alpha - beta) * 1/(rank_svd + 1)

for several (alpha, beta) settings, and shows the resulting top-5. An item
that is absent from a list simply gets no contribution from that list.
Also prints, for the top-1 item under the default, exactly how its score is
assembled term by term.

Run from repo root:  python experiments/verify_alpha_beta_demo.py [USER_ID]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_worked_example as w  # noqa: E402

SETTINGS = [
    (0.40, 0.35, "default"),
    (0.90, 0.05, "almost only User-CF"),
    (0.05, 0.90, "almost only Item-CF"),
    (0.05, 0.05, "almost only SVD (gamma=0.90)"),
    (0.34, 0.33, "roughly equal weights"),
]
TOP_N = 5


def blend(lists, alpha, beta):
    ub, ib, svd = lists
    gamma = 1 - alpha - beta
    scores = {}
    for weight, lst in ((alpha, ub), (beta, ib), (gamma, svd)):
        for rank, pid in enumerate(lst):
            scores[pid] = scores.get(pid, 0) + weight * (1 / (rank + 1))
    return scores


def main():
    uid = sys.argv[1] if len(sys.argv) > 1 else w.USER_ID
    pivot, usim, isim, pdf, pmap = w.build_models()
    n = TOP_N * 3
    ub = w.user_based_recommend(pivot, usim, uid, top_n=n)
    ib = w.item_based_recommend(pivot, isim, uid, top_n=n)
    svd = w.svd_recommend(pivot, pdf, uid, top_n=n)
    lists = (ub, ib, svd)
    nm = lambda p: pmap.get(p, {}).get("name", p)

    print(f"User {uid}: candidate lists of {n} items each (User-CF, Item-CF, SVD)\n")
    base = None
    for alpha, beta, label in SETTINGS:
        scores = blend(lists, alpha, beta)
        top = sorted(scores, key=scores.get, reverse=True)[:TOP_N]
        if base is None:
            base = top
        overlap = len(set(top) & set(base))
        print(f"alpha={alpha:.2f} beta={beta:.2f} gamma={1-alpha-beta:.2f}  [{label}]  "
              f"(overlap with default top-5: {overlap}/5)")
        for i, p in enumerate(top, 1):
            print(f"   {i}. {nm(p):<38} score={scores[p]:.4f}")
        print()

    # term-by-term breakdown of the default top-1
    alpha, beta = w.ALPHA, w.BETA
    gamma = 1 - alpha - beta
    p = base[0]
    print(f"Score breakdown for '{nm(p)}' at the default weights:")
    for label, weight, lst in (("User-CF", alpha, ub), ("Item-CF", beta, ib), ("SVD", gamma, svd)):
        if p in lst:
            r = lst.index(p) + 1
            print(f"   {label:<8} rank {r:>2}: {weight:.2f} x 1/{r} = {weight / r:.4f}")
        else:
            print(f"   {label:<8} absent from this list -> contributes 0")
    print(f"   total = {blend(lists, alpha, beta)[p]:.4f}")


if __name__ == "__main__":
    main()
