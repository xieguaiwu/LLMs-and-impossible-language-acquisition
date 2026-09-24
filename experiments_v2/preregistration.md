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
- 2026-09-19T10:15+08:00: the v2 BabyLM arm (batch 4 / seq 128 / 4010 steps) is
  RETIRED, per REDTEAM blocker #3 (regime incoherence: <2% of one epoch at
  LR 5e-5) and a fatal operational failure — the TextDataset preprocessing of
  the 8.74M-sentence train file hit the unit's 20GB MemoryMax and stalled for
  6h with zero GPU progress. Per DESIGN_V3 (frozen), the BabyLM venue for all
  cross-condition claims is the kallini_repro trainer (batch 128, seq 1024,
  3000 steps, per-sentence ppl eval). The v3 P-class grid IS the BabyLM arm.
- 2026-09-19T10:15+08:00: BabyLM source switched from the mislabeled
  Sree1994/babylm_100M HF mirror (actually 10M words; caught by the 50M-word
  guard) to the official cambridge-climb/BabyLM per-genre files — which also
  restores Kallini's 10-genre layout and per-genre eval sampling.
- 2026-09-19T10:15+08:00: BabyLM source switched from the mislabeled
  Sree1994/babylm_100M HF mirror (actually 10M words; caught by the 50M-word
  guard) to the official cambridge-climb/BabyLM per-genre files — which also
  restores Kallini's 10-genre layout and per-genre eval sampling.
- 2026-09-18: early clean-corpus results show natural ≈ reversed at the
  replication budget (test PPL 1.918 vs 1.915), which contradicts the paper's
  Exp-1 direction. Two pre-registered diagnostic arms were added (H7/H8 below)
  instead of silently interpreting the replication.
- 2026-09-20T12:45+08:00 (INCIDENT, repaired): gpu2
  `babylm_data_perturbed/babylm_parity_word/babylm_100M/simple_wikipedia_parsed.train`
  was truncated to 632,436 of 1,023,786 lines (interrupted write in the
  16:25-16:43 regeneration pass of 2026-09-19). The queue data gate checks file
  existence only, so it passed. Effect if unnoticed: the F1 treatment arm
  (`parity_word`) trains on 3.9% less data than its control (`fixed_start`),
  biased in the direction of H10; and the cpu2 LSTM arm (complete file) would
  have trained on a different data volume than the GPT-2 arm (F4c invalid).
  Repair: the single genre was regenerated on gpu2 from the frozen
  `design_v3/v3_conditions.py` (deterministic); verification = 1,023,786 lines,
  md5 8f497eefb733e944695c98b481ab7ea6 (byte-identical to the complete cpu2
  copy), pool identity parity_word = parity_tok = fixed_start = fixed_end =
  10,042,376 train lines. Truncated original quarantined (not deleted) at
  /root/kallini_data/_quarantine_20260920/. New report-only audit tool:
  experiments_v2/kallini_repro/data_integrity_check.py (pool identity per
  genre/split + cross-host md5). No training cell had consumed the truncated
  file (the parity_* block had not started).
- 2026-09-20T13:10+08:00 (design audit, pre-P-data): independent audit of the
  frozen design against the running implementation found four implementation-
  vs-design deviations and four candidate amendments; the P-class (parity_*)
  grid has produced NO data yet, so the amendments below are still pre-data.
  Deviations to record: (C1) the H7 checkpoint ladder as implemented is
  {100,300,500,1000,2000,3000,4020,6000}, not the registered
  {300,1000,2000,4000,5000,6000} (affects `auc_logstep` only); (C2) the
  implemented sentence filter is ">1 and <=350 BPE tokens" (Kallini
  filter_shuffle), not "2-200 words" as written in EXPDESIGN_V3 §1.3.1 -- the
  paper must quote the implemented rule; (C3) `final/` weights are saved for
  every cell (publish excludes them) rather than only probe-target cells;
  (C4) the H7 arm also runs `fixed_start` @6000 (not registered in F5) ->
  exploratory. Deviations NOT yet applied (owner decision pending):
  (B1) add an entropy-matched marker control `not_random` (Kallini NoReverse
  analogue) because F1 uses `fixed_start`, whose marker position entropy (0)
  does not match `parity_word`'s 50/50; (B2) the architecture axis is currently
  carried by a 1/160-budget LSTM arm -- either add a GPU token-matched LSTM arm
  (REDTEAM #4 option) or move F4 to the exploratory bucket; (B3) H9's
  pre-registered cross-class rank vector contradicts DESIGN_V3 §A.2 / REDTEAM #2
  (no raw cross-marker-family ppl ordering) and should be replaced by
  within-class orderings; (B4) seed extension as planned does not cover the
  F4 reference conditions (GPT-2 shuffle_control / reverse_full); (B5) the
  registered test-set dedup, the shared evaluation sentence-ID table,
  content-token-only (marker-masked) scoring and the marker position-entropy
  report are not implemented; (B6) `negtok` uses the base-token filter while
  the rest of class P uses the perturbed-token filter (F3 compares pools that
  differ by 0.5%); (B7) no BabyLM probe implementation exists
  (`experiments_v2/probes/probes.py` is the voided SVO version, F4), so the
  probe suite that carries claim #5 must still be written.

## 10b. Amendments registered 2026-09-20 (design audit; A P P L I E D before any class-P data existed)

Scope note: at registration time the GPT-2 grid had produced 3/42 cells, all in the
*Shuffle control family, and **no** class-P (parity_*) cell existed. Amendments that
change class-P analysis or add class-P cells are therefore still pre-data for the
families they affect; the S/R replication panel is untouched by all of them.

1. **Sentence pool v2 (base-token filter).** ``v3_conditions.write_condition`` now
   applies Kallini's ``filter_shuffle`` to the **unperturbed** sentence
   (1 < base tokens <= 350) and only then transforms it. Pool v1 filtered the
   *perturbed* token count, so every markered condition silently kept a different
   sentence set (audit A0/B5: parity_word 10,042,376 train lines vs negtok
   9,993,030; test 992,014 vs 987,793). All eight class-P conditions now share one
   sentence set by construction. The datasets were regenerated
   (``design_v3/regenerate_conditions.py --force``) and the gate now checks a
   ``.pool_version`` marker (``pool-v2-base-filter``) instead of file existence,
   because existence cannot detect a semantics change. Verified with
   ``kallini_repro/data_integrity_check.py`` on both hosts.
2. **New condition ``not_random`` (audit B1, Kallini NoReverse analogue).** Marker
   "Not" placed at sentence start or end with **exactly parity_word's marginal
   position distribution** (the same multiset of position flags, deterministically
   permuted across sentences) but independent of the sentence's parity. This gives
   F1/H10 a control whose marker-position distribution matches the treatment's,
   which ``fixed_start`` (position entropy 0) does not provide. Registered cells:
   GPT-2 seeds 0/14/41 and the GPU LSTM arm seeds 0/14/41; the ``parity_word −
   not_random`` contrast is reported inside family F1 (``F1_not_random``) alongside
   the frozen ``parity_word − fixed_start``.
3. **GPU LSTM arm at the GPT-2 token budget (audit B2).** The cpu2 LSTM arm runs at
   1/160 of the GPT-2 budget and cannot carry the architecture axis (F8). New arm:
   ``LSTM_DEVICE=cuda``, seq 1024, effective batch 128, 3000 steps,
   3000x128x1024 = 3.93e8 tokens — identical to the GPT-2 arm — on conditions
   {shuffle_control, reverse_full, parity_word, not_random} x seeds {0,14,41}, eval
   on the full 10k draw, results in ``results_lstm_gpu/`` (branch
   ``v2-results-lstm-gpu``). Optimizer settings stay the frozen per-architecture v2
   LSTM regime (AdamW 1e-3, 10% warmup, dropout 0.3, clip 5.0; EXPDESIGN_V3 §2.2).
   F4 (H12) is now a budget-matched contrast (`F4` rows carry the equal-budget
   note); the cpu2 arm remains a budget-limited diagnostic (F8 wording).
4. **Extension tier to reach the pre-registered sample sizes (audit B4).**
   seeds 53/96 for shuffle_control, reverse_full, parity_word, fixed_start,
   parity_tok, negtok (F1-F4 to n=5, the level STATS_PLAN_V3 §2 requires for
   headline claims); fixed_end at seeds 0/14/41 (F2's second control to n=3);
   H7 (6000 steps) at seeds 14/41 for shuffle_control and parity_word so F5 is no
   longer blocked at n=1; the previously unregistered ``fixed_start@6000`` cell is
   declared exploratory. Implemented in ``kallini_queue.sh`` §[4d] (the GPU LSTM arm runs first, §[4c], so F4 lands before the n=5 extension tier).
   **Execution order (2026-09-20, after the pool-v2 verification):** the class-P
   block + H7 2x run first, then the GPU equal-budget LSTM arm, then the Kallini
   S/R replication panel, then the n=5 extension tier + H7 3x. Rationale: H10/H11
   and the probe checkpoints are the paper's central contrast, the replication
   panel carries the erratum framing, and the extension tier only raises n. Total
   compute is unchanged.
   In addition, the design's *first* extension priority is now registered too:
   **H7 3x (9000 steps = 1.18e9 tokens ~= 9 epochs)** for shuffle_control and
   parity_word at seed 0, which doubles as the **Kallini token-budget fidelity
   arm** (EXPDESIGN_V3 §5, P4 ext). Grid total: 66 GPT-2 cells.
5. **Evaluation hygiene (audit B5).** ``load_eval_sentences`` now measures the pool
   duplication (**20.1% of the 987,793-sentence test pool is an exact duplicate**)
   and a sampled near-duplicate rate, and records a per-sentence id list plus an
   order fingerprint in every result JSON. The duplicate-free primary metric is
   computed **at analysis time** (``train_exp1.dedup_positions`` /
   ``analysis/v3_pipeline.py``) rather than by changing the draw, because the cpu2
   LSTM cells kept no weights and their sentence-level ppl is already fixed; this
   gives every cell — past and future — the same treatment. A content-token-only
   (marker-masked) gmean is logged alongside the primary metric for every
   checkpoint. Class-wise pools are reported exactly (S 987,793 / R 987,895 /
   P 987,793 test sentences; the residual difference is inherited from the
   upstream Kallini/Marker pipelines and is disclosed, not silently smoothed).
6. **H9 criterion amended to within-class orderings (audit B3).** The frozen
   cross-class rank vector put markered and unmarkered conditions on one raw-ppl
   axis, which DESIGN_V3 §A.2 / REDTEAM #2 forbid. H9 is now evaluated inside each
   class (S: control < local3 < local10 < evenodd < deterministic < nondeterministic;
   R: reverse_control < partial < full; P: fixed_start = fixed_end < negtok <
   parity_word < parity_tok), and the cross-class Kendall tau is still published
   descriptively with the marker-entropy caveat (``h9_replication.csv``).
7. **Probe suite implemented for BabyLM (audit B7).** ``probes/probes_babylm.py``
   replaces the voided SVO suite: branch-matched minimal-pair deltas, an asymmetric
   length cap (train <= 60 words, eval 60-200), a hidden-state probe on unmarked
   content with ``fixed_start`` as negative control, and the P4 word-vs-BPE
   dissociation probe on the disagreement subset. A code-path smoke is wired into
   the queue (§[4e], quarantined output, never a result).
8. **Analysis code (audit B8).** ``analysis/v3_pipeline.py`` implements the frozen
   families F1-F5 with Holm adjustment, paired tests, TOST, bootstrap CIs and the
   frozen verdict vocabulary, plus the H9 criterion and the deduplicated /
   content-only sensitivity columns. ``kallini_repro/grid_status.py`` is the
   registered-grid manifest used by the chain sentinel (which no longer restarts a
   completed grid).
9. **Checkpoint ladder (audit C1).** The implemented H7 ladder is
   {100,300,500,1000,2000,3000,4020,6000}; the registered
   {300,1000,2000,4000,5000,6000} was not achieved. `auc_logstep` excludes
   {100,300} as registered, so the secondary metric differs slightly from the plan;
   the primary final-checkpoint contrast is unaffected. Registered as a deviation.
10. **Sentence filter wording (audit C2).** The implemented rule is ">1 and <=350
    BPE tokens on the base sentence" (Kallini `filter_shuffle`), not "2-200 words".
    The paper must quote the implemented rule.

## 10c. Amendments registered 2026-09-20 (evening): protocol fixes + expansion arms

Authorization: the owner's standing ruling ("rigor corrections required for
methodological/academic soundness are implemented directly on the model's
recommendation") plus the explicit instruction to execute this round in full.
Data state at registration: **no confirmatory family has data under the amended
protocol** — the protocol fixes below invalidate the training regime of every cell
completed before this timestamp, and those cells are quarantined and re-run (10c-9);
every new arm below had **zero cells** when registered.

### 10c-1. n-gram statistical baseline (analysis, no training)
`analysis/ngram_baseline.py`: interpolated absolute-discounting **bigram** fitted on
the same perturbed corpora (6M-token fits, disclosed) and scored on the same frozen
10k-sentence draw with the same geometric-mean convention plus the marker-masked
content-only column. Rationale: Chomsky's critique calls an LLM "nothing but a
pattern predictor"; no experiment in this project measured that baseline, so the
"architecture-level bias" claim had no floor. **Status: implemented and run for the
7 conditions available on cpu2** (`analysis/outputs/ngram/`, results quoted in
`design_v3/FORMAL_COMPLEXITY.md` §4). Role: descriptive/exploratory floor.

### 10c-2. Capacity-matched LSTM arm — F4 redefined (**confirmatory family**)
The equal-token-budget LSTM arm (`lstm_matched`, ~40M params) is not
capacity-matched to GPT-2-small (124M), so its rows cannot carry the confirmatory
architecture claim. New arm: `LSTM_EMB=LSTM_HIDDEN=1620`, tied head ⇒
50257·1620 + 16·1620² ≈ **123.4M params (99.5 % of GPT-2-small)**, same protocol
shapes (seq 1024 / eff batch 128 / 3000 steps = 3.93e8 tokens, identical to the GPT-2
arm). **F4 is computed on this arm**; the 40M equal-budget rows are renamed
`F4_budget40M` and moved to the exploratory BH bucket. LR is frozen by a probe on
the natural condition only (3 LRs × 1 seed × 600 steps, quarantined tree, no
inferential claim; REDTEAM #4(i)). Conditions {shuffle_control, reverse_full,
parity_word} × seeds {0,14,41} = 9 cells (~40 GPU-h) in the paper-critical block,
plus seeds {53,96} = 6 cells in the stretch tier (§[4d2]) — **registered
unconditionally by owner ruling 2026-09-21**, so F4 is reported at n=5 (the
STATS_PLAN_V3 §2 headline rule); `v3_pipeline.py` selects the largest complete
paired seed set automatically and records it in the row's `seeds` column.

### 10c-3. In-process ladder probe + 2 replay cells (exploratory dynamics)
`train_exp1.py` gains `LADDER_PROBE=1`: at every evaluation checkpoint the P1
branch-matched minimal-pair delta is computed on the current weights and stored in
the cell's result JSON (`ladder_probe`). Rationale: the probe suite previously
measured only final weights, so "when was the rule (not) acquired" — the direct
experimental correspondent of the Piaget-stage narrative — was unmeasured. The
class-P blocks (§[4b], extension tier) run with this flag; the two cells finished
before it existed (`parity_word` s0, `fixed_start` s0) are **re-run** in a separate
quarantined-from-the-main-tree arm (`results_ladder_probe/`) so the panel is
seed-complete. Analysis: `v3_pipeline.run_ladder_probe_dynamics` →
`ladder_probe_dynamics.csv`, descriptive only.

### 10c-4. NoPE position-ablation arm (exploratory family **F7_nope**)
Same GPT-2 trainer with the positional embedding **zeroed and frozen** — the
semantics of Kallini's own `gpt2_no_positional_encoding_model.py` (they remove wpe;
a zeroed frozen wpe contributes the same zero vector). Estimand: the impossibility
penalty `B_m = ppl_m(cond) − ppl_m(shuffle_control)`; registered one-sided
prediction `B_gpt2 − B_nope > 0` (removing positional information reduces the
deficit, i.e. the bias is position-borne). Conditions {parity_word,
shuffle_control} × seeds {0,14,41} = 6 cells (~24 GPU-h). n=3 is deliberate: at n=2
no one-sided paired test can reach p<.05.

### 10c-5. Data-scale axis, the PoS analog (exploratory family **F8_datascale**)
Deterministic 1M/10M-token stratified subsamples of the condition's own train pool
(`design_v3/make_datascale_subsets.py`; test pool copied verbatim so the evaluation
draw is the parent's), trained at the **fixed** 3000-step budget — so the axis varies
data scarcity at fixed optimisation budget. Registered one-sided prediction:
`penalty(sub1M) > penalty(full corpus)`. Conditions {shuffle_control, parity_word,
fixed_start} × scales {sub1M, sub10M} × seeds {0,14} = 12 cells (~47 GPU-h),
exploratory/descriptive (n=2).

### 10c-6. Model-scale axis (exploratory family **F9_model_scale**)
GPT-2 **medium** (355M: n_embd 1024 / 24 layers / 16 heads) at the same token
budget, micro batch 2. Conditions {shuffle_control, parity_word, fixed_start} ×
seeds {0,14} = 6 cells (~55 GPU-h). Two-sided: the paper's own Limitations speculate
that larger models may memorise the bias away, while a larger model could equally
amplify it. Exploratory.

### 10c-7. Formal-complexity mapping (analysis/writing, no compute)
`design_v3/FORMAL_COMPLEXITY.md`: classifies every condition by the formal machinery
its transformation requires (k-local ⇒ regular; global permutation ⇒ non-regular;
reversal ⇒ anti-hierarchical but CF-closed; parity ⇒ **MOD-2 counting**, the
canonical non-regular language), reviews the transformer theory (Hahn 2020;
Merrill–Sabharwal log-precision counting), and pre-specifies six falsifiable
predictions (FC1–FC6) before the corresponding cells exist. This converts the
paper's admitted weakness ("no formal definition of impossible languages") into a
graded, testable axis.

### 10c-8. LOGO domain-transfer arm (exploratory, descriptive)
`make_datascale_subsets.py --logo`: train with the `simple_wikipedia` genre
**removed** (9/10 of the corpus), evaluate the same frozen draw; the analysis slices
per-genre perplexity and compares against the full-data model on the held-out genre.
Rationale: supports the P4 domain-dissociation probe with a data-side counterpart
(does the learned rule transfer across domains?). Conditions {shuffle_control,
parity_word} × seed 0 = 2 cells (~8 GPU-h).

### 10c-9. Protocol fixes P1/P2 + quarantine and re-run of affected cells
Two bugs in **our** reimplementation (Kallini's own training runs under NeMo and
does not contain either pattern) were found while implementing this round:

* **P1 — dropout silently disabled after the first evaluation checkpoint.**
  `evaluate_checkpoint()` sets `model.eval()` and nothing restored `model.train()`,
  so from the first checkpoint on every cell trained with dropout off (GPT-2 arm and
  both LSTM trainers).
* **P2 — gradient clipping on loss-scaled gradients.** The GPT-2 arm called
  `clip_grad_norm_(..., 1.0)` **before** `scaler.unscale_()`, i.e. it clipped the
  scaled gradients, which normalises every step to unit true norm instead of the
  registered clip@1.0 (a different optimizer regime). The LSTM arms use no scaler on
  the cpu2 path and are unaffected; the GPU LSTM arms run with AMP off.

Both are fixed in the same commit (`train_exp1.py`). Consequences: the six GPT-2
cells completed before the fix (`shuffle_control` s0, `shuffle_deterministic21` s0,
`shuffle_nondeterministic` s0, `shuffle_local3` s0, `parity_word` s0,
`fixed_start` s0) are **quarantined (moved, never deleted) and re-run** under the
fixed protocol; the re-run is automatic (their result files disappear from the arm,
so the idempotent queue re-schedules them). Effect on already-published materials:
`VISION.md`/`PROGRESS.md` numbers from those cells are marked superseded. The cpu2
LSTM arm is **left untouched** (owner ruling: do not touch that unit mid-pass; its
12 completed cells share the P1 regime uniformly and its role is budget-diagnostic).
New result JSONs record `dropout_active_all_steps` / `grad_clip_true_norm` so the two
generations of cells are distinguishable in the data itself.

### 10c-10. Queue gate hardening (row-count equality)
The class-P data gate now checks **line-count equality across the 10 genres** (train
and test), in addition to existence and the pool-version marker. A mismatch triggers
one deterministic regeneration pass and, if it persists, a hard failure (exit 9) —
the training arm must never start on an unequal pool. Rationale: audit A0 (a
37 %-truncated genre file) passed every existence check unnoticed.

### 10c-11. Infrastructure sync (manifest, sentinel, publisher)
`grid_status.py` registers all new arms and emits `pending_gpu_total`; the cpu2 chain
sentinel now prefers that field (with a fallback to the old two-key sum, so the
rolling update is safe). `publish_results.sh` (already parameterised) gains 6 new
result branches: `v2-results-lstm-gpu-capmatch`, `v2-results-nope`,
`v2-results-datascale`, `v2-results-ladder-probe`, `v2-results-logo`,
`v2-results-model-scale`. Expected totals: main GPT-2 arm 66, GPU LSTM 12, capmatch
9, NoPE 6, datascale 12, ladder-replay 2, LOGO 2, model-scale 6, cpu2 LSTM 35.
**Cost of this round ≈ 200 GPU-h** on top of the ~10.5 remaining days of the
original grid; ordering keeps every paper-critical block ahead of the exploratory
tiers (§[4d2] is last).

### 10c-12. Second compute host (Blackwell burst) — stack portability, whole-arm shards, bridge rule

**Trigger.** A second host with RTX 5090 cards became available for a bounded
window (2026-09-21). A 5090 is compute capability **sm_120**; the registered grid
ran torch 2.2.2 / CUDA 12.1, whose wheels ship **no sm_120 kernels**, so the host
necessarily runs a second numerical stack (torch >= 2.7 / cu128, or the vendor
image's torch >= 2.7 equivalent). This section registers that fact and the rules
for handling it; it changes **no** protocol parameter, no α and no family.

**Code (committed).**
* `kallini_repro/stack_compat.py` — one code path for both stacks:
  `amp_autocast()` (fp16, same semantics as the deprecated
  `torch.cuda.amp.autocast`), `grad_scaler()` (historical defaults), and
  `pin_numerics()`, which re-asserts **explicitly** the flags that were already in
  effect on the 3080 run (matmul TF32 off, cuDNN TF32 on, cuDNN benchmark off,
  deterministic off; plus `fp32_precision="ieee"` where the API exists). On the
  3080 stack these assignments are no-ops — verified: the autocast object is
  state-identical, and `torch.amp.GradScaler` does not exist on 2.2.2 so the shim
  falls back to the legacy class.
* `train_exp1.py` / `train_exp1_lstm.py` now go through the shim and write a
  `stack` block (hostname, device name + capability, torch/CUDA/cuDNN/numpy/
  transformers versions, effective precision flags) into every result JSON.
* `make_burst_shards.py` emits one TSV per GPU from the **same manifests the
  sentinel uses** (`grid_status.py`), carrying the launch recipe copied verbatim
  from `kallini_queue.sh`; `burst_shard_runner.sh` executes a shard on one GPU
  (skip-if-done, per-cell log, state TSV); `bootstrap_burst_host.sh` prepares a
  host (env with sm_120 torch, data with md5/pool verification, shards, runners);
  `bridge_check.py` performs the comparison below.

**Sharding rule.** Whole arms go to one host — no statistical family is split
across stacks (P family + H7 on one shard, architecture families on another, S/R
panel on a third, stretch tier + probe replay on the fourth). Pending lists come
from `grid_status.py`, so shards cannot silently drop or duplicate registered
cells.

**Bridge rule (pre-specified).** Before any cross-stack family is reported, at
least one **bridge cell** per stack pair is re-run on the new host with the same
condition/seed/budget, and compared against the 3080 cell:
1. `eval_fingerprint` must be **identical** (it is derived from the sampled
   sentence IDs) — otherwise the data/eval path differs and numbers are not
   compared at all;
2. worst absolute relative delta over the perplexity ladder (all + content-only,
   plus ladder-probe deltas if present):
   * **<= 1 %** ⇒ *stack-equivalent*: cross-stack contrasts may be pooled per
     family, with the venue recorded per cell;
   * **1–5 %** ⇒ *minor drift*: pool only with an explicit sensitivity note;
   * **> 5 %** ⇒ *inhomogeneous*: do **not** pool; the affected family is re-run
     on the original stack (the 3080 queue keeps its order, so nothing is lost).
3. the bridge run is registered here so it is not a post-hoc rescue: the rule and
   its thresholds are fixed before any bridge result exists.

**Precedent.** F9 (two silent protocol deviations) is why the second stack is not
trusted on faith and why the numerics flags are pinned explicitly instead of
inheriting version defaults.

**What this does not licence.** Cross-stack pooling of a family whose bridge fails,
reporting a family without its venue, or treating the burst host as a substitute
for the registered queue order.

## 10d. Deviation registered 2026-09-24: same-stack acceptance for the RTX 3090 burst host (operator decision; pending team review)

*Appended by the burst-automation agent 2026-09-24. This section is an amendment: the
text of §10c-12 above is unchanged, and its thresholds still apply as written.*

**Context.** The second host used for the burst window is an RTX 3090 node (gpu5). It
runs the **same numerical stack** as the 3080 grid — `stack_id=torch2.2.2-cu12.1`,
CUDA 12.1, cuDNN 8902, capability 8.6, identical pinned numerics flags; the two
`stack` blocks recorded in the result JSONs differ only in `device_name` (RTX 3080 vs
RTX 3090). The §10c-12 bridge cells (`parity_word`, `fixed_start`; seed 0, 3000 steps)
were re-run on gpu5 and compared with `bridge_check.py` against the 3080 new-code
reference (`/root/bridge_3080_new/results_bridge/`, code `257b07f`).

**Literal verdicts (unchanged).**
- `parity_word`: **INHOMOGENEOUS** — worst +514.42 % at step 300 (3080 103.65 → 3090
  636.87); eval fingerprints identical (`7d0e98c208871176`).
- `fixed_start`: **INHOMOGENEOUS** — worst −88.84 % at step 300 (3080 1404.78 → 3090
  156.75); eval fingerprints identical.
- Robust reading (recorded next to the literal verdict; does not enter the gate):
  `parity_word` step≥1000 worst |Δ| = 3.01 % (step 1000), final Δ = +2.00 % (step
  3000); `fixed_start` computed at verdict time and written to `/root/ss_status.md` on
  gpu5.

**Deviation.** On the operator's instruction of 2026-09-24 (user decision), gpu5 was
admitted to the grid under an explicit, recorded **same-stack acceptance**: the
literal §10c-12 verdicts above stand as written, and no pooling claim is made on their
basis. The failure is attributed to the early-training chaotic peak documented in
`current/INVESTIGATION_fixed_start_20260924.md` — the same peak appears on the 3080
under both code versions and on the 5090 pool at several seeds and steps, while the
late ladder and final values converge (parity_word: 3.01 % at step≥1000, +2.00 % at
step 3000; fixed_start step 100: 1.3 %).

**Mechanism (auditable and reversible).** The override lives in a flag file on the
burst host (`/root/kit_ss/SAME_STACK_ACCEPT`, carrying the authorisation, the evidence
and the rollback line); the admission script requires complete verdicts *and*
identical eval fingerprints, so the flag cannot bypass a data/eval-path mismatch. The
override is logged in `/root/ss_status.md` as a
`DEVIATION: same-stack acceptance per operator decision 2026-09-24` line together with
the robust reading; deleting the flag restores the original stop-on-NOT-PASS
behaviour.

**Status.** Registered for the record; **pending team review**. Any later reporting or
pooling decision for cells produced on this node can cite this section.

*Tooling note: this amendment is currently an uncommitted working-tree edit on the gpu2
checkout (`kallini_loop.sh` pulls with `git pull --ff-only` each pass and continues with
a warning if a pull is blocked by local changes). A copy of the amended file is kept at
`/root/burst/preregistration.md.bak-20260924`; reverting is
`git -C /root/llm-impossible checkout -- experiments_v2/preregistration.md`.*

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
