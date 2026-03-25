# Paper Specification (Binding Engineering Contract)

**Source**: "The Signal in the Noise: An Auditable Reliability Layer for Biomedical Text Classification"
**Conference**: ICBCB 2026
**Status**: Single source of truth. No developer may invent behaviour not described here.

---

## 1. Algorithm 1 — Conservative Token Correction with Abstention

```
Require: token t, context (p, n), vocab V, n-grams NG
1: if Protected(t) then return t             ← gates.check_protection_gates()
2: Cand ← Candidates(t, V, ED ≤ 2)          ← model._get_candidates() [LRU-cached]
3: best ← t; bestScore ← Score(t)
4: for c ∈ Cand do
5:     s ← Score(c | p, n) − α·ED(t, c)     ← model._score()
6:     if s > bestScore then best ← c; bestScore ← s
7: end for
8: if best = t then return t
9: if RealWord(t) and bestScore − Score(t) < δ_rw then return t
10: if bestScore − Score(t) < δ then return t
11: return best
```

**Code**: `model._correct_core()` is the single source of truth.
`model.correct()` and `model.correct_with_trace()` are thin wrappers.
Safety gates + margin gates (δ, δ_rw) are ALWAYS applied regardless of trace emission.

## 2. Scoring Function (Paper Eq. 2)

```
Score(c) = λ₁·log P(c) + λ₂·log P(p,c) + λ₃·log P(c,n) − α·ED(t,c)
```

**Code → Paper mapping** (all read from manifest.json via `from_artifacts()`):

| Paper | Code attribute | Property | Default | Role |
|-------|---------------|----------|---------|------|
| λ₁ | `w_uni` | `lambda_1` | 1.0 | Unigram weight |
| λ₂ | `w_bi_left` | `lambda_2` | 1.2 | Left bigram weight |
| λ₃ | `w_bi_right` | `lambda_3` | 1.0 | Right bigram weight |
| α | `w_dist` | `alpha` | 2.2 | Edit-distance penalty |
| k (unigram) | `smoothing_k` | — | 1.0 | Add-k smoothing for unigrams |
| k (bigram) | `smoothing_k_bigram` | — | 0.01 | Add-k smoothing for bigrams |
| δ | `delta` | — | 0.0 | Non-word margin threshold |
| δ_rw | `delta_rw` | — | 5.0 | Real-word margin threshold |

All scoring is unified in `_score()`.  Non-word path passes additional kwargs
(`nonword_dist_weight`, `len_ref`) that are documented as refinements within
the same function, not external additions.

## 3. N-gram Probabilities (Paper Section III-A)

Paper: "Probabilities are estimated using add-k smoothing."

**Code**: `bigram_mode = "addk"` (default, loaded from manifest):
- `_log_p_unigram(w)`: `P(w) = (count(w) + k) / (N + k·|V|)` with `smoothing_k=1.0`
- `_log_p_bigram_addk(prev, w)`: `P(w|prev) = (count(prev,w) + k) / (total(prev) + k·|V|)` with `smoothing_k_bigram=0.01`
  - Observed bigrams: add-k smoothed conditional probability
  - Unobserved bigrams: backoff to scaled unigram `α · P_uni(w)` (standard zero-count handling)
- Legacy `_log_p_bigram_backoff()` retained for `bigram_mode="backoff"` comparison

## 4. Candidate Generation (Paper Section III-A)

Paper: "we retrieve all candidates within edit distance ≤ 2 using a deletion index structure [5].
Low-frequency candidates are pruned, and repeated noisy strings are cached."

**Code** (`_generate_candidates()`):
1. **STEP 1 — Deletion-index [PRIMARY]**: `_symspell_candidates(w)` via pre-built `delete_dict`
2. **STEP 2 — Edit-based augmentation [SECONDARY]**: `edits1` + `_edits2_pruned` + `edits2` for complete recall
3. **Retrieval method tracking**: `_last_retrieval_method` = `"delete_index"` | `"combined"` | `"edits_only"`

**Cache** (`_get_candidates()`):
- True LRU via `OrderedDict` — `move_to_end()` on hit, `popitem(last=False)` on eviction
- Returns `(candidates, cache_hit, retrieval_method)` — all three stored in cache entry
- Size: 50,000 entries (configurable via manifest)

## 5. Margin Condition: `Score(c*) − Score(t) ≥ δ` (δ_rw for real-words)

## 6. Safety Gates (Paper Table I)

| Gate | Condition | Code |
|------|-----------|------|
| Short-token | len(t) ≤ 3 | `gates.is_short_token()` |
| Pattern (ALL-CAPS) | e.g. PCR, RNA | `gates.is_all_caps_abbrev()` |
| Pattern (biomed ID) | e.g. IL-6, H1N1, rs12345 | `gates.is_biomed_id()` |
| Pattern (numeric-heavy) | >50% digits/symbols | `gates.is_numeric_heavy()` |
| Real-word margin | t∈V, margin < δ_rw | `_correct_core` line 9 |
| Score margin | margin < δ | `_correct_core` line 10 |

## 7. Artifact Boundary

**Artifacts**: vocab.pkl, unigrams.pkl, bigrams.pkl, manifest.json
**Manifest config** (all loaded by `from_artifacts()`):
  delta, delta_rw, smoothing_k, smoothing_k_bigram, candidate_cache_size, bigram_mode, w_zipf
**Invariant**: same artifacts + same manifest → same output + same trace

## 8. Trace Fields (Paper Table II)

Minimal: token, position, protected, gate, candidates, scores, margin, decision
Extended: prev_token, next_token, original_score, best_candidate, best_score,
  threshold_used, threshold_type, edit_distance, artifact_version, error_type, corrected_token
Retrieval audit: retrieval_method (`"delete_index"` | `"combined"` | `"edits_only"`), candidate_cache_hit

## 9. Noise Model (Eq. 3): P(error|t) = 0.15·|t|^{-0.5}
Operators: Sub 0.50 / Del 0.25 / Ins 0.25

## 10. Four-Run Protocol: Clean → Noisy → Restored → Safety

## 11. Metrics: Macro-F1, ERR, Safety drop, Bootstrap CI (5000 default)

## 12. Classifier: TF-IDF + LogisticRegression(class_weight="balanced")

## 13. Deployment Checks: edit-rate monitoring, raw retention, artifact versioning

## 14. Target Numbers

| Metric | Value |
|--------|-------|
| Intrinsic recall | 94.61% |
| Harmful edits | 0 |
| ERR | ≈80.8% |
| Safety drop | 0.0011 |
| Bootstrap CI ΔF1 | [0.0001, 0.0122] |
| Clean F1 | 0.7732 |
| Noisy F1 | 0.7654 |
| Restored F1 | 0.7717 |
| Safety F1 | 0.7721 |
