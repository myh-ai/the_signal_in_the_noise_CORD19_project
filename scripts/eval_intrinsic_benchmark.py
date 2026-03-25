#!/usr/bin/env python
"""Intrinsic Evaluation on Curated Benchmark (2,104 cases).

Paper spec: Table VII
  - Error-fix recall (true errors) = 94.61%
  - Harmful edits on negatives = 0

This script reads synthetic_spellcheck_cases.csv, evaluates the spellchecker,
and produces metrics.json + report.csv + optional trace.jsonl.
"""

import sys
import json
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.runtime import ensure_project_root
ensure_project_root()

import pandas as pd
from config import SPELLING_ARTIFACT_DIR
from spelling.model import MedicalSpellChecker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="synthetic_spellcheck_cases.csv")
    ap.add_argument("--output_dir", default="outputs/intrinsic")
    ap.add_argument("--emit_trace", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    print(f"[INFO] Loaded {len(df)} cases from {args.input}")

    spell = MedicalSpellChecker.from_artifacts(SPELLING_ARTIFACT_DIR)

    results = []
    traces = []
    for idx, row in df.iterrows():
        noisy = str(row.get("noisy_token", row.get("token", "")))
        expected = str(row.get("expected", row.get("correct", noisy)))
        category = str(row.get("category", row.get("type", "unknown")))
        is_error = category.lower() not in ("negative", "correct", "clean")

        prev_w = str(row.get("prev_token", "")) or None
        next_w = str(row.get("next_token", "")) or None
        if prev_w == "nan": prev_w = None
        if next_w == "nan": next_w = None

        if args.emit_trace:
            corrected, err_type, trace = spell.correct_with_trace(
                noisy, position=idx, prev_word=prev_w, next_word=next_w
            )
            traces.append(trace.to_dict())
        else:
            corrected, err_type = spell.correct(noisy, prev_word=prev_w, next_word=next_w)

        corrected_l = corrected.lower()
        expected_l = expected.lower()
        noisy_l = noisy.lower()

        if is_error:
            hit = (corrected_l == expected_l)
            harmful = False
        else:
            hit = False
            harmful = (corrected_l != noisy_l)

        results.append({
            "noisy_token": noisy,
            "expected": expected,
            "corrected": corrected,
            "category": category,
            "is_error": is_error,
            "hit": hit,
            "harmful": harmful,
            "error_type": err_type if 'err_type' in dir() else "",
        })

    rdf = pd.DataFrame(results)
    errors = rdf[rdf["is_error"]]
    negatives = rdf[~rdf["is_error"]]

    recall = errors["hit"].sum() / len(errors) * 100 if len(errors) > 0 else 0.0
    harmful_count = int(negatives["harmful"].sum())
    total_errors = len(errors)
    total_negatives = len(negatives)

    metrics = {
        "total_cases": len(rdf),
        "total_errors": total_errors,
        "total_negatives": total_negatives,
        "error_fix_recall_pct": round(recall, 2),
        "harmful_edits_on_negatives": harmful_count,
        "artifact_version": spell.artifact_version,
    }

    print(f"\n{'='*50}")
    print(f" INTRINSIC BENCHMARK RESULTS")
    print(f"{'='*50}")
    print(f" Total cases:           {len(rdf)}")
    print(f" True errors:           {total_errors}")
    print(f" Negatives:             {total_negatives}")
    print(f" Error-fix recall:      {recall:.2f}%")
    print(f" Harmful edits (neg):   {harmful_count}")
    print(f"{'='*50}")

    # Save outputs
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rdf.to_csv(out_dir / "report.csv", index=False, encoding="utf-8")

    if args.emit_trace and traces:
        with open(out_dir / "trace.jsonl", "w", encoding="utf-8") as f:
            for t in traces:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"[SUCCESS] Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
