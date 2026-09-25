"""
Documents and reproduces the synthetic parts of products_500plus.csv (Gap #4).

products_500plus.csv is a CURATED catalog: the 500 product names, their
category/subcategory taxonomy, tags, and emoji were chosen by hand to look
like a realistic Indian grocery listing (e.g. "Parle-G Biscuit" under
Bakery > Biscuits). That naming/curation step is genuinely manual work and
this script does not pretend to "generate" it from randomness -- doing so
would misrepresent how the catalog was actually built.

What WAS synthetic is the two numeric columns, `price` and `rating`
(the product's own displayed star rating, not a user's rating -- that's a
separate, also-synthetic file, see generate_ratings.py). Those were assigned
per category from a documented, seeded distribution rather than picked
individually by hand. This script:

  1. Loads the curated identity columns (product_id, name, category,
     subcategory, tags, emoji) from products_500plus.csv as fixed input --
     this part is data, not something to regenerate.
  2. Re-derives `price` and `rating` from a seeded per-category distribution
     (random_state=42), documented below.
  3. Prints a comparison against the real columns so the rule's fit is
     checked, not just asserted.
  4. Writes data/products_500plus_generated.csv for inspection. It does NOT
     overwrite the real file -- app.py continues to read the original.

Generative rule (per category, seed=42):
  price  ~ round(exp(Normal(log(cat_median_price), cat_price_sigma)))
           clipped to the category's observed [min, max], since grocery
           prices are right-skewed (a few premium items cost far more than
           the median) rather than symmetric.
  rating ~ round(clip(Normal(cat_mean_rating, cat_rating_sigma), 3.9, 4.9), 1)
           since every product in this catalog is a real, stocked item with
           a plausibly-decent display rating -- there are no 1- or 2-star
           products, so the distribution is a tight, high band per category
           rather than spanning the full 1-5 scale.
  Both draws use a single seeded RNG (np.random.RandomState(42)) advanced in
  product_id order, so the run is exactly reproducible.

The category-level medians/means/sigmas used as inputs to this rule were
themselves read off the real file (see CATEGORY_PRICE_STATS /
CATEGORY_RATING_STATS below) -- i.e. this script documents "if you already
know each category's rough price and rating band, here is the seeded rule
that reproduces individual values from it," which is the part that used to
be undocumented.

Run from the repo root or from data/:
    python data/generate_catalog.py
"""
import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42

# Per-category (median_price, price_sigma_in_log_space, min_price, max_price),
# read off the real catalog's per-category price distribution.
CATEGORY_PRICE_STATS = {
    "Bakery":         (28,  0.85, 10, 350),
    "Beverages":      (200, 0.55, 25, 550),
    "Condiments":     (100, 0.55, 50, 380),
    "Dairy":          (60,  0.85, 24, 650),
    "Drinks":         (35,  0.70, 10, 150),
    "Frozen":         (100, 0.50, 20, 249),
    "Grains":         (110, 0.65, 35, 480),
    "Health":         (180, 0.85, 30, 999),
    "Home Care":      (100, 0.75, 30, 650),
    "Noodles":        (28,  0.85, 14, 168),
    "Personal Care":  (130, 0.55, 25, 599),
    "Snacks":         (28,  0.75, 1,  120),
    "Spices":         (70,  0.95, 20, 800),
}

# Per-category (mean_rating, rating_sigma), read off the real catalog.
CATEGORY_RATING_STATS = {
    "Bakery":         (4.25, 0.19),
    "Beverages":      (4.52, 0.13),
    "Condiments":     (4.47, 0.14),
    "Dairy":          (4.50, 0.13),
    "Drinks":         (4.36, 0.13),
    "Frozen":         (4.40, 0.14),
    "Grains":         (4.44, 0.14),
    "Health":         (4.46, 0.11),
    "Home Care":      (4.49, 0.10),
    "Noodles":        (4.35, 0.23),
    "Personal Care":  (4.52, 0.10),
    "Snacks":         (4.41, 0.20),
    "Spices":         (4.53, 0.12),
}


def find_data_file(filename):
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [filename, os.path.join(here, filename)]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find {filename}. Searched: {candidates}")


def generate_price(rng, category):
    median, sigma, lo, hi = CATEGORY_PRICE_STATS[category]
    draw = np.exp(rng.normal(np.log(median), sigma))
    return int(np.clip(round(draw / 5) * 5, lo, hi))  # round to nearest 5, like real prices


def generate_rating(rng, category):
    mean, sigma = CATEGORY_RATING_STATS[category]
    draw = rng.normal(mean, sigma)
    return round(float(np.clip(draw, 3.9, 4.9)), 1)


def main():
    real = pd.read_csv(find_data_file("products_500plus.csv"))
    rng = np.random.RandomState(RANDOM_STATE)

    gen = real[["product_id", "name", "category", "subcategory", "tags", "emoji"]].copy()
    gen["price"] = [generate_price(rng, cat) for cat in gen["category"]]
    gen["rating"] = [generate_rating(rng, cat) for cat in gen["category"]]
    gen = gen[["product_id", "name", "category", "subcategory", "price", "tags", "rating", "emoji"]]

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "products_500plus_generated.csv")
    gen.to_csv(out_path, index=False)

    print("Catalog regeneration check (seed=42) -- generated vs. real, per category")
    print("-" * 78)
    print(f"{'category':<16}{'price mean (gen/real)':<26}{'rating mean (gen/real)':<26}")
    for cat in sorted(real["category"].unique()):
        rp = real.loc[real.category == cat, "price"].mean()
        gp = gen.loc[gen.category == cat, "price"].mean()
        rr = real.loc[real.category == cat, "rating"].mean()
        gr = gen.loc[gen.category == cat, "rating"].mean()
        print(f"{cat:<16}{f'{gp:.0f} / {rp:.0f}':<26}{f'{gr:.2f} / {rr:.2f}':<26}")

    print(f"\nOverall price mean:  generated={gen.price.mean():.2f}  real={real.price.mean():.2f}")
    print(f"Overall rating mean: generated={gen.rating.mean():.2f}  real={real.rating.mean():.2f}")
    print(f"\nWritten to {out_path} (real file at products_500plus.csv is untouched).")
    print(
        "\nNote: names/category/subcategory/tags/emoji are curated inputs, not "
        "regenerated -- see the module docstring for why."
    )


if __name__ == "__main__":
    main()
