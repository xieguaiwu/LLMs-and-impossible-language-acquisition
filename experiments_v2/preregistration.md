# Preregistration — experiments_v2: Multi-seed Replication and Controls for
# "LLMs and Impossible Language Acquisition"

Status: DRAFT (analysis code frozen before data collection; deviating from any
clause below requires logging the deviation in section 10 at commit time).
Created: 2026-09-18. Written against the arXiv version (2602.08437) of the
paper and `statistics/EXPDESIGN.md` v1.

---

## 1. Motivation — what the original study could and could not establish

The original experiments were single-run (n=1 seed per condition) and the
published inference relied on Welch's t-tests over per-step training-loss
series (n = 141–401 per group). Per-step values within one run are strongly
serially correlated (AR(1) > 0.99), so those p-values (e.g. 1.7e-162) are
uninterpretable, as already acknowledged in the arXiv text and in
`EXPDESIGN.md` section 4.1. Additionally, three construct-validity gaps
remain open:

- **C1 (marker confound).** `parity_negation` inserts an all-caps "Not" into
  100% of sentences at a position determined by word-count parity. Lower
  natural-language loss could partly reflect marker-distribution effects
  (rare token, capitalization, position variance), not the impossibility of
  the counting rule.
- **C2 (parity domain mismatch).** Parity is computed over whitespace words,
  but the model sees BPE tokens; the rule the model must track is token-count
  parity, which is harder. The paper never states which domain is intended.
- **C3 (capacity confound).** GPT-2 small (124M) was compared with a ~40M
  LSTM with different dropout/LR regimes; the "LSTM shows no bias" claim is
  confounded with capacity and optimization.

## 2. Hypotheses (stated directionally, before data collection)

- **H1 (replication).** On SVO, GPT-2 small final loss: natural <
  parity_negation < reversed (per-seed aggregates, n=5). Expected direction
  only; magnitudes may differ from the n=1 runs.
- **H2 (marker control, C1).** If the original parity-negation deficit is
  driven by the counting rule rather than the marker, then
  fixed_start_neg ≈ natural < parity_negation. If instead
  parity_negation ≈ fixed_start_neg > natural, the deficit is a marker/
  distributional effect and the paper's interpretation must change.
- **H3 (parity domain, C2).** parity_negation_tok (BPE-token parity) shows a
  larger deficit than word-unit parity_negation.
- **H4 (special token).** parity_negation_negtok (single reserved `<NEG>`
  token) yields a deficit no smaller than "Not" — isolating capitalization/
  subword artifacts from marker presence.
- **H5 (capacity, C3).** If the LSTM null is a capacity artifact,
  capacity-matched pairs (gpt2_tiny ≈ lstm_matched) will show an
  architecture gap on SVO. If the gap disappears at matched capacity, the
  architecture claim must be weakened to a scale claim.
- **H6 (rule vs surface).** Probe predictions for a model that learned the
  parity rule (not the surface marker distribution):
  (i) mean surprisal of the marker is higher in rule-violating positions
  than rule-consistent ones (delta > 0 on ≥ 500 minimal pairs);
  (ii) a logistic probe on last-layer hidden states decodes parity class
  well above 50% chance;
  (iii) delta > 0 persists on sentences longer than any training sentence
  (length extrapolation), while a memorizing model shows delta ≈ 0 there.

## 3. Design

Factors: model {gpt2, gpt2_tiny, lstm, lstm_matched} × dataset {svo, babylm}
× condition {natural, reversed, parity_negation, fixed_start_neg,
fixed_end_neg, parity_negation_negtok, parity_negation_tok, word_shuffle} ×
seed {42..46} (5 per cell; see section 5 for the equivalence-power
extension).

- gpt2 = HuggingFace `gpt2` (124M) from scratch; gpt2_tiny = 6L/8H/512d,
  ~44M, from scratch; lstm = original 650d/2L; lstm_matched = 640d/2L, ~39M.
- Conditions are defined in `experiments_v2/data_v2/conditions.py`; every
  transformation is deterministic given the sentence.
- Seeds govern weight init, data order and dropout. All other
  hyperparameters are frozen per (model, dataset) in `training/train_lm.py`
  and logged into each run JSON.

## 4. Data and evaluation protocol

- **SVO**: 10,000 template sentences regenerated cleanly
  (`generate_svo.py`, seed 42). This fixes a latent bug in the original
  `data/generate_input.py`, which interleaved `Original: <sentence>` lines
  into the corpus that the reverse/parity transformations then treated as
  sentence content. The v2 corpus contains one plain sentence per line.
  Split: 95% train / 5% test at the sentence level (seed 42), identical
  sentence pools across conditions; the test set is perturbed per-condition
  before evaluation (matched-pair cross-condition comparison).
- **BabyLM**: raw 100M corpus from babylm.github.io; sentence split with
  spaCy when available (regex fallback), sentences of 2–200 words; 2%
  held-out with the same matched-pair protocol (`prepare_babylm.py`).
- **Budgets**: SVO 141 steps, BabyLM 401 steps (effective batch 32, block
  128) — identical to the original protocol so v2 doubles as replication.
  No early stopping; stopping rules are fixed ex ante.

## 5. Statistical analysis plan (frozen)

1. Unit of analysis: ONE SCALAR PER SEED from the run-JSON `summary` block
   (final_loss, min_loss, final_ppl, min_ppl, auc_loss, convergence_frac,
   test_loss, test_ppl). Per-step series are never used as independent
   observations.
2. Per (dataset × model × metric) family: Shapiro-Wilk per group (noting low
   power at n=5) → Levene → Welch's t when both groups pass normality,
   otherwise Mann-Whitney U.
3. Holm-Bonferroni across the pairwise comparisons within each family.
4. Cohen's d with 95% bootstrap CI (10,000 resamples over seeds).
5. Equivalence: TOST against bound d=0.8 for every comparison, so that
   null-ish results (notably the LSTM claims) are reported as *accepted
   equivalence* rather than "p > 0.05". Power note: at n=5/group a TOST at
   d=0.8 has almost no power; equivalence claims therefore additionally
   require n≈17 seeds/group for the specific cell claimed (cheap on SVO;
   we will run n=17 for the LSTM SVO cells before making any equivalence
   statement).
6. All p-values, test choices, correction status and effect sizes are
   reported in one tidy table (`holm_corrected_tests.csv`) — no selective
   reporting; non-significant and significant results appear together.

## 6. Probes (no additional training)

Run on each trained parity_negation checkpoint (`probes/probes.py`):

- P1 minimal pairs: 500 held-out pairs differing only in marker position
  (rule-obeying vs rule-violating); marker surprisal delta.
- P2 violation detection: percentage of pairs with delta > 0.
- P3 length extrapolation: pairs with more words than any training sentence.
- P4 diagnostic probe: logistic regression on last-layer hidden state →
  parity class; report test accuracy vs 50% chance.

Predictions H6(i)-(iii) are directional; P1–P4 are reported for every seed
of the parity condition (natural and reversed runs have no rule, so probes
target parity checkpoints only, with reversed checkpoints as a
negative-control where the marker exists in neither — probes on reversed
checkpoints are undefined and skipped).

## 7. Deviations from the original codebase (documented fixes)

1. `Original:` corpus pollution (section 4) — SVO corpus regenerated.
2. Loss-series t-tests replaced by per-seed aggregation + Holm.
3. Figures: seed-mean curves with 95% CI bands replace single-run overlays;
   the old per-step "t-test" figures are retired from v2 outputs.
4. Parity domain made explicit (word-unit default; token-unit as a condition).
5. Evaluation moved to condition-perturbed held-out test sets.

## 8. Compute plan

| Block | Runs | Est. GPU-h (T4) |
|---|---|---|
| SVO gpt2 3 cond × 5 seeds | 15 | ~40 |
| SVO gpt2 controls (5 cond × 5 seeds) | 25 | ~65 |
| SVO gpt2_tiny 3 × 5 | 15 | ~25 |
| SVO lstm_matched 3 × 5 | 15 | ~8 |
| SVO lstm_matched equivalence extension (n=17) | +36 | ~20 |
| BabyLM gpt2 3 × 5 | 15 | ~250–500 |
| Probes | inference only | ~2 |

Driver scripts: `run_all_svo.sh`, `run_babylm.sh` (systemd-run/tmux on the
GPU box; see experiments_v2/README.md).

## 9. Interpretation guardrails

- A replication failure of H1 is reported as such; it does not get
  re-analyzed into significance via metric shopping.
- H2 is adjudicated **before** looking at any other comparison: it decides
  whether the paper's central interpretation survives in its current form.
- The equivalence claim for LSTM (H5) is only ever stated with the n=17
  extension satisfied.

## 10. Deviation log

- 2026-09-18T23:30+08:00: the first 5 `parity_negation_negtok` cells (seeds 42-46
  started 23:09-23:25 on the GPU host) were trained WITHOUT registering `<NEG>`
  as a special token (the marker was BPE-fragmented into `< NEG >`). The bug
  was found during GPU early-signal review; those cells are invalid and were
  deleted + retrained with `add_special_tokens` + `resize_token_embeddings`.
- 2026-09-18: early clean-corpus results show natural ≈ reversed at the
  replication budget (test PPL 1.918 vs 1.915), which contradicts the paper's
  Exp-1 direction. Two pre-registered diagnostic arms were added (H7/H8 below)
  instead of silently interpreting the replication.

## 11. Post-hoc hypotheses registered on first clean-corpus evidence (H7/H8)

These were written AFTER observing the GPU early signal above (declared as
such); they are diagnostic, not confirmatory:

- **H7 (extended budget).** The natural-vs-impossible gap is an emergent
  phenomenon that requires more optimization than the original paper's
  protocol provided. Test: re-run natural/reversed/parity_negation/
  fixed_start_neg at 3× steps (4230) on the clean corpus. If the gap opens
  with budget, the original claim survives in a compute-dependent form; if
  it never opens, the Exp-1 result is corpus-artifact-driven (H8).
- **H8 (pollution artifact).** Re-implementing the original corpus format
  (each sentence duplicated as `Original: X` + `X`) restores the paper's
  numbers (natural ≈ 0.83-level loss and a natural < parity < reversed
  ordering). If confirmed, the original SVO experiments' magnitudes are a
  duplication artifact, and the paper's Exp-1 must be corrected/qualified.
