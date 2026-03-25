#!/usr/bin/env python
"""Edit-Rate Monitoring (Paper Table V: Deployment Check #1).

Computes the percentage of tokens changed per document by the reliability layer.
Alerts when edit rate exceeds a threshold (potential upstream parser failure).
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
    ap.add_argument("--input", required=True, help="CSV with text column")
    ap.add_argument("--text_col", default="clean_text")
    ap.add_argument("--threshold", type=float, default=0.10,
                    help="Alert threshold for edit rate (default: 10%%)")
    ap.add_argument("--output", default="outputs/edit_rate_report.json")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    spell = MedicalSpellChecker.from_artifacts(SPELLING_ARTIFACT_DIR)

    results = []
    alerts = 0

    for idx, row in df.iterrows():
        text = str(row[args.text_col])
        tokens = text.split()
        total = len(tokens)
        if total == 0:
            continue

        changed = 0
        for i, tok in enumerate(tokens):
            prev = tokens[i-1] if i > 0 else None
            nxt = tokens[i+1] if i < total - 1 else None
            corrected, _ = spell.correct(tok, prev_word=prev, next_word=nxt)
            if corrected.lower() != tok.lower():
                changed += 1

        edit_rate = changed / total
        alert = edit_rate > args.threshold

        if alert:
            alerts += 1
            print(f"[ALERT] doc {idx}: edit_rate={edit_rate:.4f} > {args.threshold}")

        results.append({
            "doc_id": idx,
            "total_tokens": total,
            "tokens_changed": changed,
            "edit_rate": round(edit_rate, 6),
            "alert": alert,
        })

    report = {
        "total_docs": len(results),
        "alerts": alerts,
        "threshold": args.threshold,
        "mean_edit_rate": round(sum(r["edit_rate"] for r in results) / len(results), 6) if results else 0,
        "details": results[:100],  # first 100 for brevity
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n[SUMMARY] {len(results)} docs, {alerts} alerts, mean edit_rate={report['mean_edit_rate']:.4f}")
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()
