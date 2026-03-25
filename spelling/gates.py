"""spelling.gates

Biomedical safety gates for the conservative spell-correction reliability layer.

Paper spec (Section III-F, Table II):
  - Short-token gate: protect tokens of length ≤ 3
  - Pattern gate: protect alphanumeric biomedical IDs, ALL-CAPS abbreviations, numeric-heavy tokens
  - Real-word gate: if t ∈ V, require stricter margin δ_rw before editing
  - Score margin: do not edit if bestScore − Score(t) < δ

Each gate:
  1. Short-circuits correction immediately when triggered
  2. Records in trace: protected=True, gate="<gate_name>"
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# Pre-compiled patterns for biomedical ID detection
_BIOMED_ID_RE = re.compile(
    r"^(?:"
    r"[A-Z]{1,3}[0-9]"           # IL-6, CD4, H1N1
    r"|[A-Z]{1,3}-[0-9]"         # IL-6, CD-4
    r"|rs[0-9]+"                  # rs12345 (SNP IDs)
    r"|[A-Z]{1,5}[0-9]{2,}"      # ACE2, BRCA1, TP53
    r"|[a-z]{1,3}[0-9]{2,}"      # hsa-miR, let-7
    r"|[A-Z]+[0-9]+[A-Za-z]*"    # SARS2, H5N1
    r"|[0-9]+[A-Za-z]+[0-9]*"    # 3CLpro, 5HT2A
    r")",
    re.ASCII,
)

_ALLCAPS_ABBREV_RE = re.compile(r"^[A-Z]{2,}$", re.ASCII)

# Tokens where digits+symbols comprise > 50% of characters
_NUMERIC_HEAVY_THRESHOLD = 0.5


def is_short_token(token: str) -> bool:
    """Short-token gate: protect tokens of length ≤ 3.

    Paper: 'Avoid unstable edits on low-information tokens.'
    """
    return len(token) <= 3


def is_all_caps_abbrev(token: str) -> bool:
    """Pattern gate (abbreviations): protect ALL-CAPS abbreviations.

    Examples: PCR, RNA, SARS, COVID, ARDS, ICU
    Paper: 'Preserve biomedical IDs and symbol-rich tokens.'
    """
    # Strip hyphens for tokens like "COVID-19" → check "COVID" part
    clean = token.split("-")[0] if "-" in token else token
    return bool(_ALLCAPS_ABBREV_RE.match(clean)) and len(clean) >= 2


def is_biomed_id(token: str) -> bool:
    """Pattern gate (biomedical IDs): protect alphanumeric biomedical identifiers.

    Examples: IL-6, H1N1, CD8+, rs12345, ACE2, BRCA1, TP53, hsa-miR-21
    Paper: 'Preserve biomedical IDs and symbol-rich tokens.'
    """
    # Strip common suffixes like '+' or trailing punctuation
    clean = token.rstrip("+-.,;:")
    return bool(_BIOMED_ID_RE.match(clean))


def is_numeric_heavy(token: str) -> bool:
    """Pattern gate (numeric-heavy): protect tokens where digits/symbols dominate.

    A token is numeric-heavy if > 50% of its characters are digits or symbols.
    Examples: '3.5mg', '10^6', 'p<0.05', '95%CI'
    Paper: 'Preserve biomedical IDs and symbol-rich tokens.'
    """
    if not token:
        return False
    non_alpha = sum(1 for ch in token if not ch.isalpha())
    return (non_alpha / len(token)) > _NUMERIC_HEAVY_THRESHOLD


def check_protection_gates(token: str) -> Tuple[bool, Optional[str]]:
    """Run all protection gates on a token.

    Returns:
        (protected: bool, gate_name: Optional[str])
        If protected is True, gate_name indicates which gate fired.
        Gates are checked in order; first match wins.
    """
    if is_short_token(token):
        return True, "short_token"

    if is_all_caps_abbrev(token):
        return True, "pattern_allcaps"

    if is_biomed_id(token):
        return True, "pattern_biomed_id"

    if is_numeric_heavy(token):
        return True, "pattern_numeric_heavy"

    return False, None


def check_margin_gate(best_score: float, original_score: float,
                      is_real_word: bool, delta: float, delta_rw: float) -> Tuple[bool, str]:
    """Check score-margin and real-word gates.

    Paper (Eq. 2): s(c*) − s(t) ≥ δ
    Paper (Real-word gate): if t ∈ V, require δ_rw (stricter)

    Returns:
        (should_abstain: bool, gate_name: str)
    """
    margin = best_score - original_score

    if is_real_word:
        if margin < delta_rw:
            return True, "real_word_margin"

    if margin < delta:
        return True, "score_margin"

    return False, ""
