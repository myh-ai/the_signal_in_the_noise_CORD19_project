# Reproducing Paper Results End-to-End

## Prerequisites

```bash
pip install -r requirements.txt
```

## Step 1: Prepare CORD-19 Data

```bash
python prepare_cord19.py
```

Produces `data/processed/cord19_labeled.csv` and `data/processed/cord19_corpus.txt`.

## Step 2: Split Data (Leakage Prevention)

```bash
python prepare_cord19.py --split --seed 42
```

Produces: `data/cord19_train.csv`, `data/cord19_dev.csv`, `data/cord19_test.csv`.

## Step 3: Build Spelling Artifacts (from train only)

```bash
python spelling/build_spelling_model.py --input data/cord19_train.csv
```

Produces versioned artifacts in `spelling/artifacts/` with `manifest.json` (SHA256 hashes, version ID).

## Step 4: Train Downstream Classifier (Paper Mode)

```bash
python classification/train_classifier.py --paper_mode
```

Forces: LogisticRegression + class_weight="balanced". Saves `topic_classifier_logreg.joblib` + `classifier_manifest.json`.

## Step 5: Synthesize Noise (Paper Model)

```bash
python scripts/synthesize_corruptions.py \
  --input test.csv \
  --output data/test_synth_paper.csv \
  --paper_noise \
  --seed 42
```

Uses exact paper noise model: P(error|t) = 0.15·|t|^(-0.5), weights [Sub 0.50, Del 0.25, Ins 0.25].
Output columns: doc_id, clean_text, noisy_text, label, noise_seed, noise_config_id

## Step 6: Run Four-Run Evaluation

```bash
python scripts/evaluate_impact_fast_v2.py \
  --input data/test_synth_paper.csv \
  --max_rows 10000 \
  --bootstrap 5000 \
  --seed 42 \
  --output_report outputs/eval_paper/metrics.json \
  --output_rows outputs/eval_paper/predictions.csv \
  --emit_trace
```

Outputs:
- `outputs/eval_paper/metrics.json` — all metrics + bootstrap CI
- `outputs/eval_paper/predictions.csv` — per-row predictions with label_flip flags
- `outputs/eval_paper/trace_restored.jsonl` — token-level audit trace

## Step 7: Run Intrinsic Benchmark

```bash
python scripts/eval_intrinsic_benchmark.py \
  --input synthetic_spellcheck_cases.csv \
  --output_dir outputs/intrinsic/ \
  --emit_trace
```

## Step 8: Constrained Parameter Tuning

```bash
python scripts/tune_spellchecker_constrained.py \
  --benchmark synthetic_spellcheck_cases.csv \
  --output outputs/tuning/results.json
```

## Step 9: Deployment Checks

```bash
# Edit-rate monitoring
python scripts/monitor_edit_rate.py --input test.csv --text_col abstract

# Raw + corrected export
python scripts/export_raw_and_corrected.py --input test.csv --text_col abstract
```

## Step 10: Verify Target Numbers

Expected from `outputs/eval_paper/metrics.json`:

| Metric       | Expected |
|--------------|----------|
| Clean F1     | 0.7732   |
| Noisy F1     | 0.7654   |
| Restored F1  | 0.7717   |
| Safety F1    | 0.7721   |
| ERR          | ~80.45%  |
| ΔF1 CI       | [0.0001, 0.0122] |

Expected from `outputs/intrinsic/metrics.json`:

| Metric               | Expected |
|----------------------|----------|
| Error-fix recall     | 94.61%   |
| Harmful edits        | 0        |
