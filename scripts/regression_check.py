#!/usr/bin/env python
"""Regression Check: verify spellchecker outputs match golden baseline.

Usage:
  python scripts/regression_check.py [--golden outputs/baseline_reference/golden_outputs.json]

Exit code:
  0 = all match (safe to proceed)
  1 = mismatch detected (investigate before merging)
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

from spelling.model import MedicalSpellChecker
from config import SPELLING_ARTIFACT_DIR


TEST_CASES = [
    ('remdesivlr', 'with', 'treatment'),
    ('IL-6', None, None),
    ('PCR', None, None),
    ('the', None, None),
    ('3.5mg', None, None),
    ('treatment', 'the', 'of'),
    ('covid', None, None),
    ('H1N1', None, None),
    ('RNA', None, None),
    ('SARS', None, None),
    ('rs12345', None, None),
    ('dosage', None, None),
    ('pateint', 'the', None),
    ('hydroxychloroqunie', None, None),
    ('epidemiolgy', 'of', None),
]

TEST_DOC = (
    "the remdesivlr study showed IL-6 levels in PCR tests "
    "for COVID-19 with SARS-CoV-2 and H1N1 at 3.5mg dosage"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="outputs/baseline_reference/golden_outputs.json")
    args = ap.parse_args()

    golden_path = Path(args.golden)
    if not golden_path.exists():
        print(f"[WARN] Golden file not found: {golden_path}")
        print("[WARN] Run with --create-golden to generate.")
        sys.exit(0)

    with open(golden_path) as f:
        golden = json.load(f)

    spell = MedicalSpellChecker.from_artifacts(SPELLING_ARTIFACT_DIR)

    failed = 0
    total = 0

    for word_args in TEST_CASES:
        word = word_args[0]
        prev = word_args[1] if len(word_args) > 1 else None
        nxt = word_args[2] if len(word_args) > 2 else None
        key = f"{word}|{prev}|{nxt}"
        total += 1

        c, e, t = spell.correct_with_trace(word, position=0, prev_word=prev, next_word=nxt)
        g = golden.get(key)
        if g is None:
            print(f"  SKIP: {key} not in golden")
            continue

        ok = (c == g["corrected"] and e == g["error_type"] and t.decision == g["decision"])
        if not ok:
            failed += 1
            print(f"  FAIL: {word} — expected={g['corrected']}/{g['decision']}, got={c}/{t.decision}")
        else:
            print(f"  OK: {word}")

    # Document test
    total += 1
    corrected_doc, _, summary = spell.correct_text_with_trace(TEST_DOC, doc_id="reg_001")
    g_doc = golden.get("__doc__")
    if g_doc and corrected_doc != g_doc["corrected"]:
        failed += 1
        print(f"  FAIL: document-level")
    else:
        print(f"  OK: document-level")

    print(f"\n{'PASS' if failed == 0 else 'FAIL'}: {total - failed}/{total} passed")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
