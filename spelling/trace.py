"""spelling.trace

Token-level decision trace for the conservative spell-correction reliability layer.

Implements the audit-trace schema from Paper Table III (Minimal Audit-Trace Fields):
  token, position, protected, gate, candidates, scores, margin, decision

Extended fields provide additional context for debugging and reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class CandidateRecord:
    """A single correction candidate with its score and distance."""
    word: str
    dist: int
    score: float


@dataclass
class TraceRecord:
    """Token-level audit trace (Paper Table III + extended fields).

    Minimal fields (paper spec):
        token, position, protected, gate, candidates, scores, margin, decision

    Extended fields (recommended for full reproducibility):
        prev_token, next_token, original_score, best_candidate, best_score,
        threshold_used, threshold_type, edit_distance, artifact_version, error_type
    """

    # --- Minimal (Paper Table III) ---
    token: str
    position: int
    protected: bool = False
    gate: Optional[str] = None
    candidates: List[CandidateRecord] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    margin: float = 0.0
    decision: str = "abstain"  # "apply" | "abstain"

    # --- Extended ---
    prev_token: Optional[str] = None
    next_token: Optional[str] = None
    original_score: float = 0.0
    best_candidate: Optional[str] = None
    best_score: float = 0.0
    threshold_used: float = 0.0
    threshold_type: str = "delta"  # "delta" | "delta_rw"
    edit_distance: int = 0
    artifact_version: str = ""
    error_type: str = ""  # "Correct" | "Non-word error" | "Real-word error" | "Unknown (OOV)"
    corrected_token: Optional[str] = None

    # --- Retrieval audit (Paper Section III-A: deletion-index + caching) ---
    retrieval_method: str = ""  # "delete_index" | "fallback_edits" | "combined"
    candidate_cache_hit: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON output."""
        d = asdict(self)
        # Convert CandidateRecord list to plain dicts
        d["candidates"] = [asdict(c) for c in self.candidates]
        return d

    def to_json(self) -> str:
        """Serialize to single-line JSON for JSONL output."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class DocSummary:
    """Document-level summary statistics for audit."""
    doc_id: str = ""
    total_tokens: int = 0
    tokens_changed: int = 0
    tokens_protected: int = 0
    tokens_abstained: int = 0
    edit_rate: float = 0.0
    prediction_changed: bool = False
    artifact_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["_type"] = "doc_summary"
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)
