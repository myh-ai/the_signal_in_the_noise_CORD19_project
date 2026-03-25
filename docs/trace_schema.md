# Audit Trace Schema

## Token-Level Trace Record (JSONL format)

Each line in a `.jsonl` trace file is a JSON object with these fields:

### Minimal Fields (Paper Table III)

```json
{
  "token": "remdesivlr",
  "position": 42,
  "protected": false,
  "gate": null,
  "candidates": [
    {"word": "remdesivir", "dist": 1, "score": -18.32}
  ],
  "scores": {"original": -24.55, "best": -18.32},
  "margin": 6.23,
  "decision": "apply"
}
```

### Extended Fields

```json
{
  "prev_token": "with",
  "next_token": "treatment",
  "original_score": -24.55,
  "best_candidate": "remdesivir",
  "best_score": -18.32,
  "threshold_used": 0.0,
  "threshold_type": "delta",
  "edit_distance": 1,
  "artifact_version": "v1_abc123def456",
  "error_type": "Non-word error",
  "corrected_token": "remdesivir"
}
```

### Gate-Protected Example

```json
{
  "token": "IL-6",
  "position": 15,
  "protected": true,
  "gate": "pattern_biomed_id",
  "candidates": [],
  "scores": {},
  "margin": 0.0,
  "decision": "abstain"
}
```

## Code Mapping

- `spelling/trace.py` → `TraceRecord` dataclass (all fields)
- `spelling/trace.py` → `CandidateRecord` (word, dist, score)
- `spelling/trace.py` → `DocSummary` (document-level stats)
- `spelling/model.py` → `correct_with_trace()` (emits TraceRecord per token)
- `spelling/model.py` → `correct_text_with_trace()` (emits list + DocSummary)

## Document-Level Summary

```json
{
  "_type": "doc_summary",
  "doc_id": "cord19_00042",
  "total_tokens": 156,
  "tokens_changed": 3,
  "tokens_protected": 12,
  "tokens_abstained": 5,
  "edit_rate": 0.0192,
  "prediction_changed": false,
  "artifact_version": "v1_abc123def456"
}
```

## File Naming Convention

- `trace_restored.jsonl` — traces from Restored run (noisy → corrected)
- `trace_safety.jsonl` — traces from Safety run (clean → corrected)
