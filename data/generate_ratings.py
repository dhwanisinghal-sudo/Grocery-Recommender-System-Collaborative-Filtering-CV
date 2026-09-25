"""
Documents and reproduces user_ratings.csv from a seeded synthetic process
(Gap #4). Unlike the product catalog (see generate_catalog.py), this file
is fully synthetic end to end -- there are no real users behind it -- so
this script regenerates the whole thing, not just a subset of columns.

Generative story (random_state=42, advanced per user in user_id order):

  1. Ratings per user: each user rates a random integer number of products,
     n ~ Uniform{30, ..., 60}. This matches the real file's per-user count
     range (30-60, mean 45.3) and is what keeps overall sparsity near 90.9%
     (150 users x ~45 ratings / (150 x 500 products) ~= 9%).

  2. Category affinity: each user gets a per-category preference vector
     drawn from a Dirichlet(alpha=0.5 across the 13 categories). A low alpha
     produces "peaky" affinities -- most users end up with 2-3 favorite
     categories rather than rating uniformly across all 13 -- which is what
     produces the real file's uneven per-product popularity (some products
     rated by only 4 users, others by 26; mean 13.6, std ~4).

  3. Product selection: for a given user, the probability of a product being
     among the ones they rate is proportional to
         (their affinity for its category) x (the product's own catalog
         rating, from products_500plus.csv, as a popularity proxy).
     n products are then sampled without replacement using those weights.
     This is why higher-rated catalog items tend to accumulate more user
     ratings, same as in the real file.

  4. Rating value: a two-part mixture, not a single bell curve, because the
     real file's histogram jumps sharply at 3.5 (few ratings in 2.8-3.4,
     many more from 3.5 up) rather than following one smooth distribution:
       - with probability 0.76 ("liked"):     rating ~ Uniform(3.5, 5.0)
       - with probability 0.24 ("not liked"): rating ~ Uniform(1.5, 3.4)
     both rounded to the nearest 0.1, matching the real file's 0.1-step
     scale. The 0.76/0.24 split and the two bounds were read directly off
     the real file's rating histogram.

This reproduces the real file's summary statistics closely (mean rating,
sparsity, per-user and per-product rating counts) without claiming to
recover the exact original random draws -- see the printed comparison below.

Run from the repo root or from data/:
    python data/generate_ratings.py
"""
import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_USERS = 150
MIN_RATINGS_PER_USER = 30
MAX_RATINGS_PER_USER = 60
DIRICHLET_ALPHA = 0.5
P_LIKE = 0.76


def find_data_file(filename):
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [filename, os.path.join(here, filename)]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find {filename}. Searched: {candidates}")


def main():
    products = pd.read_csv(find_data_file("products_500plus.csv"))
    real = pd.read_csv(find_data_file("user_ratings.csv"))

    categories = products["category"].unique().tolist()
    cat_of_product = products.set_index("product_id")["category"].to_dict()
    catalog_rating_of = products.set_index("product_id")["rating"].to_dict()
    product_ids = products["product_id"].tolist()

    rng = np.random.RandomState(RANDOM_STATE)
    rows = []
    for i in range(N_USERS):
        uid = f"U{i + 1:03d}"
        n = rng.randint(MIN_RATINGS_PER_USER, MAX_RATINGS_PER_USER + 1)

        affinity = rng.dirichlet(np.full(len(categories), DIRICHLET_ALPHA))
        cat_affinity = dict(zip(categories, affinity))

        weights = np.array([
            cat_affinity[cat_of_product[pid]] * catalog_rating_of[pid]
            for pid in product_ids
        ])
        weights = weights / weights.sum()

        chosen = rng.choice(product_ids, size=min(n, len(product_ids)), replace=False, p=weights)
        for pid in chosen:
            if rng.random() < P_LIKE:
                rating = rng.uniform(3.5, 5.0)
            else:
                rating = rng.uniform(1.5, 3.4)
            rows.append((uid, pid, round(float(rating), 1)))

    gen = pd.DataFrame(rows, columns=["user_id", "product_id", "rating"])

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_ratings_generated.csv")
    gen.to_csv(out_path, index=False)

    def stats(df):
        by_user = df.groupby("user_id").size()
        by_product = df.groupby("product_id").size()
        return {
            "n_ratings": len(df),
            "mean_rating": df.rating.mean(),
            "sparsity": 1 - len(df) / (N_USERS * len(product_ids)),
            "avg_ratings_per_user": by_user.mean(),
            "avg_ratings_per_product": by_product.mean(),
            "ratings_per_product_range": (by_product.min(), by_product.max()),
        }

    g, r = stats(gen), stats(real)
    print("Ratings regeneration check (seed=42) -- generated vs. real")
    print("-" * 60)
    print(f"{'metric':<28}{'generated':<16}{'real':<16}")
    print(f"{'n_ratings':<28}{g['n_ratings']:<16}{r['n_ratings']:<16}")
    print(f"{'mean_rating':<28}{g['mean_rating']:<16.3f}{r['mean_rating']:<16.3f}")
    print(f"{'sparsity':<28}{g['sparsity']:<16.4f}{r['sparsity']:<16.4f}")
    print(f"{'avg_ratings_per_user':<28}{g['avg_ratings_per_user']:<16.2f}{r['avg_ratings_per_user']:<16.2f}")
    print(f"{'avg_ratings_per_product':<28}{g['avg_ratings_per_product']:<16.2f}{r['avg_ratings_per_product']:<16.2f}")
    print(f"{'ratings_per_product_range':<28}{str(g['ratings_per_product_range']):<16}{str(r['ratings_per_product_range']):<16}")
    print(f"\nWritten to {out_path} (real file at user_ratings.csv is untouched).")


if __name__ == "__main__":
    main()
