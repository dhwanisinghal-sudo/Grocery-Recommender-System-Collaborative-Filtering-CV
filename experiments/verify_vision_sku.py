"""
Per-SKU vision accuracy (small, hand-verified subset).

verify_vision_accuracy.py measures CATEGORY-level accuracy on 260 images. This
script measures whether the pipeline picks the exact catalog product, using
data/vision_test_set/labels_sku.csv, which lists only the photos that show a
product that really exists in data/products_500plus.csv (23 images). Photos of
products that are not in the catalog can't have a correct product_id, so they
are not part of this test.

Runs the same path the app uses:
    classify_image()  ->  find_products_from_tags()  ->  ranked product_ids

Metrics per image:
    top1  : the first returned product_id is an acceptable id
    top3  : an acceptable id is in the first 3
    top6  : an acceptable id is anywhere in the returned list (the app shows up to 6)
Some products appear twice in the catalog under different ids (e.g. Frooti
P063/P249), so labels_sku.csv lists every acceptable id separated by '|'.

Usage (from repo root):
    python experiments/verify_vision_sku.py
Optional: GEMINI_API_KEY / HF_API_TOKEN in .streamlit/secrets.toml to exercise
the API stages. Without them, images OCR can't read fall through to the color
heuristic, which can't identify a specific product. Report OCR-resolved and
API-resolved images separately when you compare runs.

Writes experiments/vision_sku_results.csv.
"""
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import app  # noqa: E402  (Streamlit script; bare-mode import works)

HERE = os.path.dirname(os.path.abspath(__file__))
TEST_SET_DIR = os.path.join(HERE, "..", "data", "vision_test_set")
LABELS_SKU = os.path.join(TEST_SET_DIR, "labels_sku.csv")
OUT_CSV = os.path.join(HERE, "vision_sku_results.csv")
SLEEP = float(os.environ.get("SKU_SLEEP", "2.5"))  # pace API calls


def main():
    with open(LABELS_SKU, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} SKU-labeled images\n")

    results = []
    for i, r in enumerate(rows, 1):
        rel = r["filename"]
        acceptable = set(r["acceptable_product_ids"].split("|"))
        with open(os.path.join(TEST_SET_DIR, rel), "rb") as fh:
            image_bytes = fh.read()
        try:
            tags, method, _dbg = app.classify_image(image_bytes)
        except Exception as ex:  # keep going; record the failure
            print(f"  [{i}/{len(rows)}] {rel}: classify_image raised {ex!r}")
            tags, method = [], "error"
        time.sleep(SLEEP if method not in ("📝 OCR Text Detection",) else 0)

        matched = app.find_products_from_tags(tags or [], app.product_map)
        top1 = bool(matched) and matched[0] in acceptable
        top3 = any(p in acceptable for p in matched[:3])
        top6 = any(p in acceptable for p in matched[:6])
        results.append({
            "filename": rel, "expected": r["acceptable_product_ids"],
            "product_name": r["product_name"], "photo_type": r["photo_type"],
            "stage": method, "returned": "|".join(matched),
            "top1": top1, "top3": top3, "top6": top6,
        })
        print(f"  [{i}/{len(rows)}] {r['product_name']:<26} {method:<24} "
              f"top1={top1!s:<5} returned={matched[:3]}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    def summarize(label, subset):
        n = len(subset)
        if not n:
            return
        t1 = sum(x["top1"] for x in subset)
        t3 = sum(x["top3"] for x in subset)
        t6 = sum(x["top6"] for x in subset)
        print(f"  {label:<34} n={n:<3} top1 {t1}/{n} ({t1/n*100:.0f}%)  "
              f"top3 {t3}/{n} ({t3/n*100:.0f}%)  top6 {t6}/{n} ({t6/n*100:.0f}%)")

    print("\n=== Per-SKU accuracy ===")
    summarize("ALL", results)
    for pt in sorted({x["photo_type"] for x in results}):
        summarize(f"photo_type = {pt}", [x for x in results if x["photo_type"] == pt])
    for st in sorted({x["stage"] for x in results}):
        summarize(f"stage = {st}", [x for x in results if x["stage"] == st])
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
