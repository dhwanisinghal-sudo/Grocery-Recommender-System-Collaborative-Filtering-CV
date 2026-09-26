"""
Verifies Gap #1: the vision pipeline has never had a measured accuracy number.

Runs the real classify_image() cascade from app.py (OCR -> Gemini -> HF ->
color fallback) on data/vision_test_set/ -- 260 labeled images, 20 per
catalog category, 13 categories -- and reports, per stage:
  - how many images that stage resolved (i.e. was the one that answered)
  - what fraction of those it got right

IMPORTANT — what "correct" means here:
  The test images are representative category photos (e.g. 20 general
  "Dairy" product photos), not phone photos of the exact 500 SKUs in
  data/products_500plus.csv. There is no per-SKU ground truth available,
  so this script measures **category-level** accuracy: does the top
  matched product (via find_products_from_tags()) belong to the same
  catalog category as the test image's folder? This is a coarser, honest
  substitute for per-product accuracy -- report it as such, not as
  per-SKU accuracy.

IMPORTANT — Gemini / HuggingFace stages:
  These call external APIs and require GEMINI_API_KEY / HF_API_TOKEN in
  .streamlit/secrets.toml. If neither key is set, every image falls
  through OCR -> (Gemini: no key) -> (HF: no key) -> color fallback, so
  this run's numbers describe the OCR and color-fallback stages only.
  Add real keys and re-run to also get Gemini/HF accuracy.
  (Building this script surfaced a real bug: classify_image_gemini() /
  classify_image_hf() read st.secrets without a try/except, which raises
  -- not "no key" -- when .streamlit/secrets.toml doesn't exist at all,
  crashing the whole pipeline instead of falling back. Fixed in app.py by
  wrapping those two reads in try/except.)

Run from the repo root:
    python experiments/verify_vision_accuracy.py
"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# app.py is a Streamlit script; importing it outside `streamlit run` just
# logs "missing ScriptRunContext" warnings and runs in bare mode -- the
# pure functions and module-level data (product_map, etc.) all work fine.
import app  # noqa: E402

TEST_SET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "vision_test_set")
LABELS_CSV = os.path.join(TEST_SET_DIR, "labels.csv")


def load_labels():
    rows = []
    with open(LABELS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def main():
    rows = load_labels()
    print(f"Loaded {len(rows)} labeled images across "
          f"{len(sorted(set(r['category'] for r in rows)))} categories.\n")

    # stage -> {"n": total resolved by this stage, "correct": how many right}
    stage_stats = {}
    per_category = {}
    results = []

    t0 = time.time()
    for i, row in enumerate(rows, 1):
        rel_path = row["filename"]
        true_category = row["category"]
        img_path = os.path.join(TEST_SET_DIR, rel_path)

        with open(img_path, "rb") as f:
            image_bytes = f.read()

        try:
            tags, method, _debug = app.classify_image(image_bytes)
        except Exception as ex:
            method = "ERROR"
            tags = []
            print(f"  [{i}/{len(rows)}] {rel_path}: classify_image raised {ex!r}")

        matched = app.find_products_from_tags(tags or [], app.product_map)
        pred_category = app.product_map.get(matched[0], {}).get("category") if matched else None
        correct = (pred_category == true_category)

        stage_stats.setdefault(method, {"n": 0, "correct": 0})
        stage_stats[method]["n"] += 1
        stage_stats[method]["correct"] += int(correct)

        per_category.setdefault(true_category, {"n": 0, "correct": 0})
        per_category[true_category]["n"] += 1
        per_category[true_category]["correct"] += int(correct)

        results.append({
            "filename": rel_path,
            "true_category": true_category,
            "stage": method,
            "predicted_category": pred_category or "",
            "correct": correct,
        })

        if i % 50 == 0:
            print(f"  ...{i}/{len(rows)} images processed ({time.time()-t0:.0f}s elapsed)")

    # ---- write per-image results ----
    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vision_accuracy_results.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "true_category", "stage", "predicted_category", "correct"])
        w.writeheader()
        w.writerows(results)

    # ---- report ----
    total_n = len(rows)
    total_correct = sum(r["correct"] for r in results)

    print("\n" + "=" * 70)
    print("Table — Vision Pipeline Accuracy by Stage (category-level)")
    print("=" * 70)
    print(f"{'Stage':<24}{'Resolved':<12}{'% of images':<14}{'Accuracy':<10}")
    for stage, s in sorted(stage_stats.items(), key=lambda kv: -kv[1]["n"]):
        pct_of_total = s["n"] / total_n * 100
        acc = (s["correct"] / s["n"] * 100) if s["n"] else 0.0
        print(f"{stage:<24}{s['n']:<12}{pct_of_total:<14.1f}{acc:<10.1f}")

    print("-" * 70)
    print(f"{'OVERALL':<24}{total_n:<12}{'100.0':<14}{total_correct/total_n*100:<10.1f}")

    print("\nPer-category accuracy:")
    for cat, s in sorted(per_category.items()):
        acc = s["correct"] / s["n"] * 100 if s["n"] else 0.0
        print(f"  {cat:<16} {s['correct']:>2}/{s['n']:<3} ({acc:.0f}%)")

    print(f"\nFull per-image results written to {out_csv}")


if __name__ == "__main__":
    main()
