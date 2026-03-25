#!/usr/bin/env python
"""Export Raw + Corrected Text (Paper Table V: Deployment Check #2).

Retains raw text alongside corrected text, artifact_version, and run_id
for traceability and rollback support.
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

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
    ap.add_argument("--input", required=True)
    ap.add_argument("--text_col", default="clean_text")
    ap.add_argument("--output", default="outputs/raw_and_corrected.csv")
    ap.add_argument("--run_id", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    spell = MedicalSpellChecker.from_artifacts(SPELLING_ARTIFACT_DIR)

    run_id = args.run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    rows = []
    for idx, row in df.iterrows():
        text = str(row[args.text_col])
        corrected, _, summary = spell.correct_text_with_trace(text)
        rows.append({
            "doc_id": idx,
            "raw_text": text,
            "corrected_text": corrected,
            "edit_rate": summary.edit_rate,
            "tokens_changed": summary.tokens_changed,
            "artifact_version": spell.artifact_version,
            "run_id": run_id,
        })

    out_df = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[SUCCESS] Exported {len(rows)} docs to {out_path}")
    print(f"[INFO] Artifact version: {spell.artifact_version} | Run ID: {run_id}")


if __name__ == "__main__":
    main()
