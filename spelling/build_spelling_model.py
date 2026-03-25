"""spelling.build_spelling_model

Build spelling artifacts (vocab, unigrams, bigrams) from a CORD-19 corpus or
a training CSV.

Paper spec (Section IV — Artifact Boundary):
  - Artifacts are built from the TRAINING partition ONLY.
  - Each build produces a manifest.json with SHA-256 hashes for reproducibility.
  - Artifacts MUST NOT use dev/test data.

Usage:
  # From corpus file (legacy):
  python spelling/build_spelling_model.py

  # From training CSV (paper-compliant):
  python spelling/build_spelling_model.py --input data/cord19_train.csv --text_col full_text
"""

import sys
import os
import json
import hashlib
import pickle
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# --- Runtime bootstrap ---
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.runtime import ensure_project_root
ensure_project_root()

from utils.preprocess import preprocess_for_spelling

from config import (
    CORPUS_PATH,
    SPELLING_ARTIFACT_DIR,
    VOCAB_PATH,
    UNIGRAMS_PATH,
    BIGRAMS_PATH,
    SPELLING_MANIFEST_PATH,
)

MIN_FREQ = 3  # ignore extremely rare terms


def sha256_file(path: str) -> str:
    """Compute SHA-256 hex digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_from_lines(lines, source_description: str = "corpus"):
    """Build vocab, unigrams, bigrams from an iterable of text lines."""
    unigram_counts = Counter()
    bigram_counts = defaultdict(Counter)
    total_tokens = 0

    for line_idx, line in enumerate(lines, start=1):
        tokens = preprocess_for_spelling(line)
        if not tokens:
            continue

        unigram_counts.update(tokens)
        total_tokens += len(tokens)

        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i + 1]
            bigram_counts[w1][w2] += 1

        if line_idx % 5000 == 0:
            print(f"[INFO] Processed {line_idx} lines from {source_description}...")

    vocab = {w for w, c in unigram_counts.items() if c >= MIN_FREQ}

    print(f"\n[METRIC] Total tokens: {total_tokens:,}")
    print(f"[METRIC] Vocab size (freq >= {MIN_FREQ}): {len(vocab):,}")
    print(f"[METRIC] Bigram entries: {sum(len(v) for v in bigram_counts.values()):,}")

    return vocab, unigram_counts, bigram_counts, total_tokens


def _get_git_commit() -> str:
    """Best-effort git commit hash."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return ""


def main():
    ap = argparse.ArgumentParser(description="Build spelling artifacts from CORD-19 data.")
    ap.add_argument("--input", default=None,
                    help="Training CSV file (paper-compliant). If omitted, reads from corpus file.")
    ap.add_argument("--text_col", default="full_text",
                    help="Text column in CSV (default: full_text). Also tries 'clean_text', 'abstract'.")
    ap.add_argument("--version_tag", default=None,
                    help="Optional version tag (default: auto-generated timestamp).")
    args = ap.parse_args()

    import pandas as pd

    source_hash = ""
    source_path = ""

    if args.input:
        # Paper-compliant: build from training CSV
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        source_path = str(input_path)
        source_hash = sha256_file(source_path)
        source_type = "train_partition"

        df = pd.read_csv(input_path)
        # Try multiple column names
        text_col = None
        for col_name in [args.text_col, "full_text", "clean_text", "abstract", "text"]:
            if col_name in df.columns:
                text_col = col_name
                break
        if text_col is None:
            raise ValueError(f"No text column found. Available: {list(df.columns)}")

        print(f"[INFO] Building artifacts from {input_path} (column: {text_col})")
        print(f"[INFO] Source hash: {source_hash[:16]}...")
        print(f"[INFO] Source type: TRAIN PARTITION (paper-compliant, no leakage)")

        lines = df[text_col].dropna().astype(str).tolist()
        vocab, unigram_counts, bigram_counts, total_tokens = build_from_lines(
            lines, source_description=str(input_path)
        )
    else:
        # Legacy: build from corpus file
        print("[WARN] Building from general corpus (NOT train-only).")
        print("[WARN] For paper compliance, use: --input data/cord19_train.csv --text_col full_text")
        if not os.path.exists(CORPUS_PATH):
            raise FileNotFoundError(
                f"Corpus file not found: {CORPUS_PATH}. "
                "Please run prepare_cord19.py first or use --input."
            )
        source_path = str(CORPUS_PATH)
        source_hash = sha256_file(source_path)
        source_type = "corpus_general"

        print(f"[INFO] Reading corpus from: {CORPUS_PATH}")

        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()

        vocab, unigram_counts, bigram_counts, total_tokens = build_from_lines(
            lines, source_description=str(CORPUS_PATH)
        )

    # Save artifacts
    os.makedirs(SPELLING_ARTIFACT_DIR, exist_ok=True)

    with open(VOCAB_PATH, "wb") as f:
        pickle.dump(vocab, f)
    with open(UNIGRAMS_PATH, "wb") as f:
        pickle.dump(unigram_counts, f)
    with open(BIGRAMS_PATH, "wb") as f:
        pickle.dump(bigram_counts, f)

    # Generate version tag
    version_tag = args.version_tag or f"v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Compute artifact hashes
    vocab_hash = sha256_file(str(VOCAB_PATH))
    unigrams_hash = sha256_file(str(UNIGRAMS_PATH))
    bigrams_hash = sha256_file(str(BIGRAMS_PATH))

    # Build manifest (Paper: artifact versioning as deployment check)
    manifest = {
        "version": version_tag,
        "timestamp": datetime.now().isoformat(),
        "source": {
            "path": source_path,
            "sha256": source_hash,
            "type": source_type,  # "train_partition" (paper) or "corpus_general" (legacy)
        },
        "artifacts": {
            "vocab.pkl": {"sha256": vocab_hash, "vocab_size": len(vocab)},
            "unigrams.pkl": {"sha256": unigrams_hash, "total_tokens": total_tokens},
            "bigrams.pkl": {"sha256": bigrams_hash,
                           "entries": sum(len(v) for v in bigram_counts.values())},
        },
        "config": {
            "min_freq": MIN_FREQ,
        },
        "git_commit": _get_git_commit(),
    }

    SPELLING_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n[SUCCESS] Spelling artifacts saved in: {SPELLING_ARTIFACT_DIR}")
    print(f"[SUCCESS] Manifest: {SPELLING_MANIFEST_PATH}")
    print(f"[SUCCESS] Version: {version_tag}")


if __name__ == "__main__":
    main()
