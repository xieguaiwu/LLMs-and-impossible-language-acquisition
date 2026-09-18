# experiments_v2 — Multi-seed replication, controls, and probes

This directory implements the methodological strengthening of the paper:

1. **n≥5 seeds per condition** with per-seed statistical aggregates
   (replacing time-series t-tests) + Holm-Bonferroni correction + TOST
   equivalence testing for null claims.
2. **Marker controls** for the parity-negation condition
   (`fixed_start_neg`, `fixed_end_neg`), a **special-token variant**
   (`parity_negation_negtok`, single reserved `<NEG>`), a **token-unit
   parity variant** (`parity_negation_tok`), and Kallini-style
   `word_shuffle` as a reference condition.
3. **Capacity-matched architecture pair** (`gpt2_tiny` ≈44M vs
   `lstm_matched` ≈39M) alongside the original subjects.
4. **Behavioral probes** for the parity rule: minimal pairs, violation
   detection, length extrapolation, diagnostic hidden-state probe.
5. **Matched-pair held-out evaluation**: one deterministic sentence split
   shared by all conditions; test PPL computed on per-condition perturbed
   copies of the same held-out sentences.

Preregistration: `preregistration.md` (read before interpreting any result).

## Layout

```
data_v2/
  generate_svo.py        clean SVO corpus (fixes the Original: pollution bug)
  conditions.py          all conditions + deterministic split
  prepare_babylm.py      BabyLM sentence-split + per-condition transforms
training/
  models.py              LSTMLM (tied embeddings) + param accounting
  train_lm.py            unified entry: --model --dataset --condition --seed
  metrics.py             per-step logging + per-run scalar summary (JSON)
probes/
  probes.py              minimal pairs / surprisal / extrapolation / probe
analysis/
  aggregate_seeds.py     runs -> all_runs.csv / per_seed_summary.csv
  stats_tests.py         Shapiro/Levene/Welch/MWU + Holm + d-CI + TOST
  plots.py               seed-mean curves with 95% CI + per-seed box plots
tests/
  test_all.py            CPU smoke tests (16 checks, no GPU needed)
run_all_svo.sh           full SVO suite (45+ runs + probes + stats)
run_babylm.sh            BabyLM 5-seed replication
results/                 run JSONs + aggregated CSVs + plots (created on run)
```

## Quick start (GPU box)

```bash
git clone https://github.com/xieguaiwu/LLMs-and-impossible-language-acquisition.git
cd LLMs-and-impossible-language-acquisition
pip install -r requirements.txt   # + scipy for analysis

# CPU smoke tests first (optional but recommended):
python3 experiments_v2/tests/test_all.py

# Full SVO suite (~2-4 days on one T4; ~6 h on one A100):
tmux new -s svo
bash experiments_v2/run_all_svo.sh

# BabyLM (place raw corpus, then):
python3 experiments_v2/data_v2/prepare_babylm.py --raw <babylm_100M_txt_dir>
bash experiments_v2/run_babylm.sh            # RUN_LSTM=1 to add lstm_matched
```

One-off runs:

```bash
python3 experiments_v2/training/train_lm.py \
  --model gpt2_tiny --dataset svo --condition fixed_start_neg --seed 43

python3 experiments_v2/probes/probes.py \
  --model-dir experiments_v2/results/svo/gpt2/parity_negation_seed42 \
  --model gpt2 --n-pairs 500 --extrapolation --probe-diagnostic

python3 experiments_v2/analysis/aggregate_seeds.py
python3 experiments_v2/analysis/stats_tests.py    # -> holm_corrected_tests.csv
python3 experiments_v2/analysis/plots.py
```

## Reading the outputs

- `results/aggregated/holm_corrected_tests.csv` — one row per comparison:
  test choice, raw + Holm p, Cohen's d with bootstrap CI, TOST equivalence.
  `significant_holm` is the only significance flag to report.
- `results/svo/*/*/probe_report.json` — rule-learning evidence (delta > 0,
  probe accuracy ≫ chance, extrapolation delta > 0).
- Per-run `training_metrics.json` is backward-compatible with the original
  statistics scripts (they will keep working) and adds the `summary` block
  that v2 analysis consumes.

## What each new condition is for

| Condition | Purpose |
|---|---|
| `fixed_start_neg` / `fixed_end_neg` | marker-distribution control (C1): same "Not" marker, no parity rule |
| `parity_negation_negtok` | removes capitalization/subword artifacts; marker is one reserved token |
| `parity_negation_tok` | parity over BPE tokens — the domain the model actually sees (C2) |
| `word_shuffle` | Kallini et al. reference condition for cross-paper calibration |
| `gpt2_tiny` / `lstm_matched` | capacity-matched architecture pair (C3) |
