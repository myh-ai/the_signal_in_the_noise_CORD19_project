#!/usr/bin/env python
"""Constrained Parameter Tuning for Spellchecker.

Paper spec (Section IV-C):
  Parameters {lambda1, lambda2, lambda3, alpha, delta, delta_rw} are selected
  on a held-out development split using a constrained objective:
  - Maximize intrinsic recall and downstream recovery
  - Subject to: ZERO harmful edits on negative cases (strict safety constraint)
  - Coarse grids for thresholds and weight ratios (auditable tuning)
"""

import sys
import json
import argparse
import itertools
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.runtime import ensure_project_root
ensure_project_root()

import pandas as pd
from config import SPELLING_ARTIFACT_DIR
from spelling.model import MedicalSpellChecker


def evaluate_config(spell, df):
    """Evaluate a spellchecker config on the intrinsic benchmark."""
    hits = 0
    total_errors = 0
    harmful = 0
    total_negatives = 0

    for _, row in df.iterrows():
        noisy = str(row.get("noisy_token", row.get("token", "")))
        expected = str(row.get("expected", row.get("correct", noisy)))
        category = str(row.get("category", row.get("type", "unknown")))
        is_error = category.lower() not in ("negative", "correct", "clean")

        prev_w = str(row.get("prev_token", "")) or None
        next_w = str(row.get("next_token", "")) or None
        if prev_w == "nan": prev_w = None
        if next_w == "nan": next_w = None

        corrected, _ = spell.correct(noisy, prev_word=prev_w, next_word=next_w)

        if is_error:
            total_errors += 1
            if corrected.lower() == expected.lower():
                hits += 1
        else:
            total_negatives += 1
            if corrected.lower() != noisy.lower():
                harmful += 1

    recall = hits / total_errors * 100 if total_errors > 0 else 0.0
    return {"recall": recall, "harmful": harmful, "total_errors": total_errors, "total_negatives": total_negatives}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default="synthetic_spellcheck_cases.csv")
    ap.add_argument("--output", default="outputs/tuning/results.json")
    args = ap.parse_args()

    df = pd.read_csv(args.benchmark)
    print(f"[INFO] Loaded {len(df)} benchmark cases")

    # Coarse grid (Paper: transparent, auditable)
    # Includes δ and δ_rw (Paper Eq. 2 — margin gates)
    grid = {
        "delta": [0.0, 0.5, 1.0],
        "delta_rw": [3.0, 5.0, 7.0],
        "w_dist": [1.5, 2.2, 3.0],
        "w_bi_left": [0.8, 1.2, 1.5],
    }

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"[INFO] Testing {len(combos)} configurations...")

    results = []
    best_config = None
    best_recall = -1

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))

        spell = MedicalSpellChecker.from_artifacts(SPELLING_ARTIFACT_DIR)
        for k, v in params.items():
            setattr(spell, k, v)

        metrics = evaluate_config(spell, df)

        # STRICT SAFETY CONSTRAINT: reject if harmful > 0
        if metrics["harmful"] > 0:
            status = "REJECTED (harmful > 0)"
        else:
            status = "ACCEPTED"
            if metrics["recall"] > best_recall:
                best_recall = metrics["recall"]
                best_config = params.copy()
                best_config["_recall"] = metrics["recall"]
                best_config["_harmful"] = metrics["harmful"]

        results.append({**params, **metrics, "status": status})

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(combos)}] best_recall={best_recall:.2f}%")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "grid": grid,
        "total_configs": len(combos),
        "accepted_configs": sum(1 for r in results if r["status"] == "ACCEPTED"),
        "best_config": best_config,
        "all_results": results,
    }
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*50}")
    print(f" CONSTRAINED TUNING RESULTS")
    print(f"{'='*50}")
    print(f" Total configs:    {len(combos)}")
    print(f" Accepted (safe):  {output['accepted_configs']}")
    print(f" Best recall:      {best_recall:.2f}%")
    print(f" Best config:      {best_config}")
    print(f" Saved to:         {out_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
