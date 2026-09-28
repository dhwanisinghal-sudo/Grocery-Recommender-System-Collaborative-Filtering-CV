"""
Runs EACH vision stage on its own over all 260 labeled images, instead of the
cascade in verify_vision_accuracy.py.

Why: in the cascade, a stage only sees the images the earlier stages gave up on.
In the reference run Hugging Face saw 2 images and the color heuristic saw none,
so their standalone accuracy was unmeasured; and in the first run their numbers
(24.2%, 6.1%) were on the hard leftover images only. Here every stage sees the
same 260 images, so the stages can be compared directly.

Stages (all from app.py):
  ocr    classify_image_ocr()   Tesseract + keyword dictionary (offline)
  color  color_fallback()       hue/brightness/texture rules   (offline)
  gemini classify_image_gemini() needs GEMINI_API_KEY in .streamlit/secrets.toml
  hf     classify_image_hf()    needs HF_API_TOKEN   in .streamlit/secrets.toml

"Correct" = the top product from find_products_from_tags() is in the image's
true catalog category (category-level, same definition as the cascade script).
A stage that returns no tags counts as *unresolved*; unresolved images count as
wrong in the "over all 260" column.

Each stage writes experiments/vision_isolated_<stage>.csv and can be resumed:
images already in the file are skipped. The summary reads whatever stage files
exist, so offline and keyed stages can be run at different times / machines.

Usage (repo root):
    python experiments/verify_vision_stages_isolated.py --stages ocr,color
    python experiments/verify_vision_stages_isolated.py --stages gemini,hf
    python experiments/verify_vision_stages_isolated.py --summary-only
"""
import argparse
import csv
import io
import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import app  # noqa: E402
from PIL import Image  # noqa: E402

TEST_SET_DIR = os.path.join(HERE, "..", "data", "vision_test_set")
LABELS_CSV = os.path.join(TEST_SET_DIR, "labels.csv")
FIELDS = ["filename", "true_category", "resolved", "predicted_category", "correct"]
STAGE_NAMES = {"ocr": "OCR", "gemini": "Gemini", "hf": "HuggingFace", "color": "Color heuristic"}
ORDER = ["ocr", "gemini", "hf", "color"]


def run_stage(stage, image_bytes, retries):
    if stage == "color":
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return app.color_fallback(img)
    fn = {"ocr": app.classify_image_ocr, "gemini": app.classify_image_gemini,
          "hf": app.classify_image_hf}[stage]
    for attempt in range(1 + (retries if stage in ("gemini", "hf") else 0)):
        tags, _err, _dbg = fn(image_bytes)
        if tags:
            return tags
        if stage in ("gemini", "hf") and attempt < retries:
            time.sleep(5)  # transient 503 / rate limit: wait and retry
    return None


def out_path(stage):
    return os.path.join(HERE, f"vision_isolated_{stage}.csv")


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run(stages, sleep, retries):
    labels = read_rows(LABELS_CSV)
    for stage in stages:
        path = out_path(stage)
        done = {r["filename"] for r in read_rows(path)} if os.path.exists(path) else set()
        new_file = not os.path.exists(path)
        todo = [r for r in labels if r["filename"] not in done]
        print(f"[{stage}] {len(done)} done, {len(todo)} to run")
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                w.writeheader()
            for i, row in enumerate(todo, 1):
                with open(os.path.join(TEST_SET_DIR, row["filename"]), "rb") as fh:
                    data = fh.read()
                try:
                    tags = run_stage(stage, data, retries)
                except Exception as ex:
                    print(f"  {row['filename']}: {stage} raised {ex!r}")
                    tags = None
                pred = ""
                if tags:
                    matched = app.find_products_from_tags(tags, app.product_map)
                    pred = app.product_map.get(matched[0], {}).get("category", "") if matched else ""
                w.writerow({"filename": row["filename"], "true_category": row["category"],
                            "resolved": bool(tags), "predicted_category": pred,
                            "correct": bool(tags) and pred == row["category"]})
                f.flush()
                if stage in ("gemini", "hf"):
                    time.sleep(sleep)
                if i % 25 == 0:
                    print(f"  [{stage}] {i}/{len(todo)}", flush=True)


def summary():
    data = {s: read_rows(out_path(s)) for s in ORDER if os.path.exists(out_path(s))}
    if not data:
        print("No vision_isolated_*.csv files found. Run with --stages first.")
        return
    hard = None
    if "ocr" in data:
        hard = {r["filename"] for r in data["ocr"] if r["resolved"] != "True"}
    n_cat = len({r["true_category"] for r in next(iter(data.values()))})
    print("\n" + "=" * 100)
    print("Each stage run ALONE on the same images (category-level accuracy)")
    print("=" * 100)
    hdr = f"{'Stage':<18}{'Images':>7}{'Resolved':>10}{'Acc if resolved':>17}{'Acc over all':>14}"
    if hard is not None:
        hdr += f"{'On OCR-unresolved subset (n=' + str(len(hard)) + ')':>44}"
    print(hdr)
    for s, rows in data.items():
        n = len(rows)
        res = [r for r in rows if r["resolved"] == "True"]
        cor = sum(r["correct"] == "True" for r in rows)
        acc_res = f"{100 * sum(r['correct'] == 'True' for r in res) / len(res):.1f}%" if res else "n/a"
        line = f"{STAGE_NAMES[s]:<18}{n:>7}{len(res):>10}{acc_res:>17}{100 * cor / n:>13.1f}%"
        if hard is not None:
            sub = [r for r in rows if r["filename"] in hard]
            sres = [r for r in sub if r["resolved"] == "True"]
            scor = sum(r["correct"] == "True" for r in sub)
            line += f"{'resolved ' + str(len(sres)) + ', correct ' + str(scor) + ' (' + format(100 * scor / len(sub), '.1f') + '%)':>44}" if sub else ""
        print(line)
    print("-" * 100)
    print(f"Chance level for {n_cat} balanced categories: {100 / n_cat:.1f}%")
    for s, rows in data.items():
        top = Counter(r["predicted_category"] or "(none)" for r in rows).most_common(3)
        print(f"  {STAGE_NAMES[s]:<16} most frequent predictions: " + ", ".join(f"{c} x{k}" for c, k in top))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="", help="comma list of: ocr,gemini,hf,color")
    ap.add_argument("--sleep", type=float, default=2.5, help="seconds between API calls")
    ap.add_argument("--retries", type=int, default=2, help="API retries when a stage returns nothing")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    if a.stages and not a.summary_only:
        run([s.strip() for s in a.stages.split(",") if s.strip()], a.sleep, a.retries)
    summary()
