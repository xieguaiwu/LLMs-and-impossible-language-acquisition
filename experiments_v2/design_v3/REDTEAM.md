# REDTEAM — v3 design attack (Momus review, 2026-09-19)

Scope: v3 plan (BabyLM-100M primary; Kallini-style classes with controls; held-out perturbed
geometric-mean ppl at checkpoint ladder; GPT-2-small 124M vs LSTM ~39M; parity-negation family
with fixed-position/special-token controls; probe suite; replication-vs-3x extended arm).
Reviewed: KALLINI_DESIGN_NOTES.md, preregistration.md (incl. §10–11 deviations), the arXiv tex,
and the v2 code (train_lm.py, conditions.py, prepare_babylm.py, models.py, probes.py,
kallini_repro/train_exp1.py). Ranked by severity: findings that invalidate published numbers
and planned inferences first, interpretation/framing last.

---

## 1. BLOCKER — The revision's own evidence refutes the paper's published headline numbers, and the design does not yet adjudicate this.
(a) The paper's Exp-1 core (SVO: natural 0.83-level loss; parity 1.96x, reversed 2.19–2.25x
natural) is triple-contaminated: the `Original:` duplication bug (fixed in v2, H8 arm), the
"Not"-marker confound (H2), and n=1 seeds; the clean replication shows natural ≈ reversed
(1.918 vs 1.915) and marker-controls *below* natural (fixed 1.677/1.677, parity 1.668, negtok
1.606). (b) A hostile reviewer reads v3's own preregistered deviations and concludes the
revision's experiments contradict the paper being revised: if v3 runs and stays silent about
this, the paper is self-refuting; if it reports only favorable v3 cells, that is selective
reporting by construction. (c) Mitigation (zero GPU cost): restructure the revision as
erratum-first — report H8's polluted-vs-clean side-by-side, retire the t-test figures still
referenced in the tex (`t_test_loss_comparison.png`, `t_test_baby_perplexity_comparison.png`),
drop the AUC metric and the "loss+ppl is more objective than Kallini" claim (ppl = e^loss is
the same information), and make BabyLM the only venue for cross-condition ordering claims.

## 2. BLOCKER — Marker-entropy confound poisons geometric-mean ppl compared across differently transformed corpora; the planned estimand is wrong as stated.
(a) Geometric-mean test ppl of condition C under model-C is dominated by each test set's
surface statistics: parity/fixed conditions add one position-predictable marker token to 100%
of sentences, so they are *a priori* lower-ppl — v2 already shows markered conditions beating
natural (1.61–1.68 < 1.92). Cross-condition geometric means are only interpretable as deltas
against a marker-matched control, and reversal/shuffle conditions have no marker at all.
(b) This is the mechanism behind threat 1 and will silently regenerate the same artifact in
v3: "parity < natural" would again be read as "parity is easier" when it is marker entropy.
Every headline comparison in the v3 plan is affected, including the checkpoint-ladder figures.
(c) Mitigation: freeze the estimand as within-family control deltas — parity − fixed_start,
parity_tok − fixed controls, negtok − parity; compare Reversed-with-R-marker only against
NoReverse-with-R-marker (Kallini's design); additionally report content-token-only scoring
(mask the marker positions in eval) and publish per-condition marker position-entropy; never
state a natural-vs-markered raw ppl ordering.

## 3. BLOCKER — The budget axis is incoherent: BabyLM runs ~0.016 epochs, SVO runs ~10 epochs; "acquisition" claims are being read off two uninterpretable regimes.
(a) train_lm.py BUDGETS (line 46): BabyLM gpt2 = 4010 steps × batch 4 × seq 128 ≈ 2.05M
token-visits over a ~130M-token corpus (<2% of one epoch), at LR 5e-5 — 12x below Kallini's
6e-4 with 512x smaller batch and 8x shorter sequences; SVO = 1410 × 512 ≈ 0.72M visits over a
~70k-token corpus ≈ 10+ epochs (pure memorization regime; the 3x arm's test ppl 1.92→2.79
proves it). The paper's text says "trained on BabyLM ~100M words" — the model never completes
one pass. (b) At 0.016 epochs, condition differences reflect warmup/optimizer transients, not
learnability; at 10 epochs, differences are memorization asymmetries. The H7 extended arm (3x
of an already-wrong budget) measures neither regime; kallini_repro (batch 128, ~3 epochs,
6e-4) is the only run so far in a sane regime. (c) Mitigation: define budgets in *epochs*, not
steps, for all v3 cells (e.g., ladder to 3 and 11 epochs as primary points); for the 3080, run
the primary contrast set (natural, word_shuffle, Reversed+marker-control, parity, fixed_start,
fixed_end) at Kallini-equivalent hyperparameters via kallini_repro machinery (~0.39B tokens ≈
5–8 h/run on 3080; 6 cond × 3 seeds ≈ 100–145 GPU-h ≈ 5–6 days) and relegate the batch-4/seq-128
arm to SVO-scale diagnostics with an explicit "toy" label in the paper.

## 4. BLOCKER — The LSTM null result is undetermined between "no bias" and "undertrained/misconfigured"; as designed it cannot be adjudicated.
(a) The LSTM gets a different optimization regime everywhere: dropout 0.3 vs 0.1, LR 1e-3 vs
5e-5, weight-decay 1e-5 vs 0.01, grad-clip 5.0 vs 1.0, per-line 63-token truncation vs 128-token
chunking (train_lm.py:248), EOS-padded batches whose pad positions contribute to loss
(models.py:52–57 sets ignore_index=-100 but labels are never -100), and 2000 steps × batch 32
× block 64 ≈ 4.1M visits — a different epoch count from every GPT-2 arm. The prereg itself
concedes TOST at n=5, d=0.8 has "almost no power", yet the paper's conclusion "LSTM shows no
inductive bias" is the paper's architectural headline. (b) A hostile ML reviewer needs one
sentence: "your LSTM never saw a converged epoch under a tuned schedule, so the null is
uninterpretable" — and the paper's only novel empirical axis collapses with it (H5's
capacity-vs-architecture fork is then decided by a bug-shaped confound, not evidence).
(c) Mitigation: (i) tune LR per family on the natural condition only (3 LRs × 2 seeds, cheap),
then freeze; (ii) run LSTM on BabyLM at epochs-matched budget with cuDNN + packed sequences
(batch 128, seq 256, ~5–6 h/run; 3 primary conditions × 3 seeds ≈ 50 GPU-h); (iii) mask pads
in LSTMLM loss (one-line fix); (iv) keep the n=17 TOST extension for SVO cells only, and word
all BabyLM LSTM claims as "no detectable gap at this budget (CIs reported)", never "no bias".

## 5. BLOCKER — "Reversed" without a marker control is not comparable to Kallini's FullReverse; every replication claim built on it is unfounded.
(a) Kallini's *Reverse class inserts marker token R in both conditions and control at matched
positions (entropy control); our `reversed` (conditions.py rule_reverse) is bare reversal,
vocab 50257 vs their 50258, produced by a different perturber, evaluated by a different
protocol. (b) Any sentence of the form "we replicate Kallini's Reverse continuum / our reversed
condition corresponds to their FullReverse" is false as stated, and worse, the bare-reversal
gap is uninterpretable: without the NoReverse+R control we cannot separate the order-destroying
effect from any marker-related term — precisely the confound the field (and our own H2) criticizes
in the parity condition. (c) Mitigation: (i) designate kallini_repro (their perturb.py verbatim,
their per-sentence ppl function) as the *only* arm allowed to make replication claims;
(ii) relabel our bare `reversed` as "Reverse-bare (no marker)" and add one marker-matched pair
on BabyLM — Reversed+fixed-R vs NoReverse+R — 2 extra conditions × 3 seeds ≈ 2 × 3 × 5–8 h ≈
30–48 GPU-h; (iii) in the tex, replace "the reversed condition" language with class-internal
contrasts only.

## 6. BLOCKER — The v3 evaluation plan (held-out perturbed geometric-mean ppl per checkpoint) is not what the code computes; cross-condition chunk misalignment and unmasked pads leak into every number.
(a) eval_loss_gpt2 (train_lm.py:198) concatenates the whole test file and chunks at 128 tokens
with no sentence isolation or EOS: predictions cross sentence boundaries, and chunk boundaries
shift across conditions when a transformation changes token count (parity adds ±1 token per
sentence; negtok changes vocab), so the "identical sentence pool" guarantee breaks at the token
level; the CLM collator leaves pad(=EOS) labels unmasked. Kallini's quantity is per-sentence ppl
geometric-mean over 10k perturbed sentences — implemented correctly only in kallini_repro.
(b) Every cross-condition comparison inherits boundary/pad noise, and the checkpoint ladder
would be built on a metric that does not match the prereg's own definition — a protocol-vs-code
contradiction a reviewer can demonstrate from the repo. (c) Mitigation: port kallini_repro's
per-sentence perplexity + EOS-joined packing into the v3 trainer for both architectures;
evaluate ladder checkpoints in-process (no 600 GB checkpoint store; keep only final checkpoint
for probes, ~0.5 GB/run); re-evaluate all claims' tables on the fixed protocol before unblinding.

## 7. WARNING→BLOCKING — Train/test contamination and test-set selection: redundant splits, template degeneracy, no dev set.
(a) BabyLM held-out = 2% random split of the same corpus with no dedup (prepare_babylm.py:74):
BabyLM is transcript-dominated and near-duplicate-rich, so "held-out" sentences have train-set
twins; the SVO pool is template-generated so a held-out sentence is a near-copy of train items
of the same template, and word-count parity is nearly a deterministic function of the template —
the parity rule is largely *lexicalized* on SVO; there is no dev set anywhere, so any checkpoint
or condition highlighted from the ladder is selected on the test set. (b) Absolute "learnability"
numbers and the extended-budget (H7) conclusions are contaminated; relative matched-pair deltas
survive duplication but only if no selection-on-test occurs; a hostile reviewer will ask for the
duplication rate and will reject any "gap opens at step X" claim made from the ladder.
(c) Mitigation: MinHash-dedup both BabyLM train and eval pools (report near-dup rate); switch
BabyLM held-out to the official BabyLM dev/test files where possible; carve 1% dev for all
selection (checkpoint choice, LR choice); pre-commit that ladder curves are reported whole
(no "best checkpoint" extraction); on SVO, either drop headline claims or add template-disjoint
splits and report the template-parity contingency table.

## 8. WARNING — The probe suite has three construct-validity defects that would each sink the "rule learned" conclusion.
(a) P3 length extrapolation is empty by construction: prepare_babylm.py:34/66 and the SVO
generator cap sentences at ≤200 words, so no test sentence exceeds max train length
(probes.py make_extrapolation_pairs can return n=0; its train_pool even uses all sentences,
line 248, not the train split); P4's logistic probe on SVO decodes template identity/length,
not parity (parity ⟂ template on SVO; also probe target is learnable from marker position
trivially on markered inputs); P1's delta conflates rule knowledge with positional priors
because obeying vs violating positions are 0 vs N within each pair, asymmetric by construction.
(b) The paper's most quotable philosophical claim ("the model learned the counting rule, not the
surface marker") rests entirely on these probes; three independent confounds make the claim
unsupported even if deltas come out positive. (c) Mitigation: (i) build P3 from BabyLM with an
asymmetric length cap (train ≤ 60 words; eval draws 60–200), and synthesize pairs if needed;
(ii) restrict P4 to prefix hidden states *before* the marker on unmarked content, plus
fixed_start checkpoints as negative control, run on BabyLM (SVO probe results void);
(iii) for P1/P2 report length- and parity-branch-matched deltas (obeying-first vs violating-last
AND obeying-last vs violating-first) so position and rule are decorrelated.

## 9. WARNING — Double-edge: the paper's philosophical frame converts its own architecture finding into a Chomskyan result.
(a) The planned finding — GPT-2/transformer shows a natural-vs-impossible gap, LSTM does not —
is exactly "UG-as-architecture": the transformer's initial weights are as innate as any posited
faculty, and the paper itself already concedes the LSTM null "is compatible with the predictions
of Chomsky's critique" (tex, Exp 4 discussion). Current v2 signals sharpen the edge: at the
replication budget the transformer does NOT separate natural from reversed, and markered
conditions beat natural — i.e., bias tracks linear-position and marker statistics, not
hierarchy, which is Chomsky's "pattern predictor" premise, not its refutation. (b) A Chomskyan
reviewer flips the paper into a confirmation of Chomsky and mocks the "functionalist paradigm
shift" as rhetoric riding on someone else's best paper; a neutral reader is left unsure what the
experiments were supposed to show. (c) Mitigation: pre-register (already sketched in prereg §9)
an outcome-to-interpretation matrix and print it in the paper; scope the claim to "LMs possess
architectural, not UG-specific, inductive biases; Chomsky's premise (a) is false only in its
strongest form"; add an explicit UG-as-architecture discussion paragraph; move Ryle/Piaget/
Halliday material to a clearly-labeled philosophical commentary section decoupled from the
experimental evidence.

## 10. FRAMING — Kallini-lineage novelty deficit: without pre-committed deltas, v3 is a partial replication of an ACL best paper plus an essay; compute reality forces the choice now.
(a) On one 3080 the full Kallini-protocol matrix (8 conditions × 2 architectures × 2 datasets ×
5 seeds + 3x arm) is infeasible (hundreds to ~800 GPU-h); the feasible matrix must be pruned ex
ante, and the pruning decision IS the novelty statement. The deltas that survive scrutiny:
(1) architecture axis at matched capacity (Kallini: GPT-2 only) — conditional on fixing #4/#5/#7;
(2) marker-control family for counting rules (fixed_start/fixed_end/negtok) — a genuine
generalization of Kallini's Reverse-marker principle to the counting-rule class;
(3) budget-dependence as an epoch ladder on BabyLM (Kallini: single budget) — replaces the
broken 3x-on-SVO design (H7's test channel is dead);
(4) the H8 artifact diagnosis + corrected SVO protocol — a methodological caution for the genre;
(5) length-extrapolation probe — new, after #8 fixes. NOT deltas: loss+ppl "dual indicators"
(spurious), AUC, per-step t-tests (retired), "more Chomskyan transformations" rhetoric.
(b) If v3 ships without these deltas frozen in the prereg, a reviewer's one-liner — "Exp 1 of
Kallini at 1/750th compute, plus philosophy" — is accurate and fatal. (c) Mitigation: amend the
preregistration with a frozen reduced matrix (BabyLM primary: 6–8 conditions × gpt2-small × 3
seeds at epochs-matched budget ≈ 100–150 GPU-h; SVO full matrix incl. tiny/lstm_matched on the
cheap 128-block arm on CPU; extended-budget arm only on the 2–3 primary contrasts × 2 seeds,
expressed as additional ladder points, not a separate protocol), commit the deviation log, and
rewrite the tex contributions section to exactly the four deltas above.

---

## Deltas that justify the paper (vs Kallini et al. 2024) — verdict
- Justifying if executed cleanly: matched-capacity architecture axis; marker-control family for
  counting-rule languages; epoch-ladder budget-dependence on BabyLM; H8/corpus-artifact
  correction (with transparent erratum framing); length-extrapolation probe (after fix).
- Not justifying / must be cut: "loss+ppl more objective than Kallini"; AUC; per-step t-tests;
  SVO headline ratios; "our linear transformations are more Chomskyan than shuffle" rhetoric
  (no formal definition of impossible is given — the paper admits this in Limitations).
- Replication-fidelity evidence lives in kallini_repro only; everything else is a new experiment
  and must be labeled as such.

## Compute note (single RTX-3080 10 GB + 4-core CPU)
- kallini_repro GPT-2-small at batch-128/seq-1024/3000 steps ≈ 0.39B tokens ≈ 5–8 h/run (scaled
  from the measured ~95 s/run @ 2.05M visits): BabyLM primary 6–8 cond × 3 seeds ≈ 5–6 days
  continuous GPU. Adopt Kallini stability flags in this arm only (already in train_exp1.py:207).
- LSTM BabyLM epochs-matched: cuDNN + packed seq 256 + batch 128 ≈ 5–6 h/run; 3 × 3 cells ≈
  50 GPU-h. Mask pads first (models.py:52–57).
- CPU box: SVO/tiny/LSTM 128-block arm, probe inference, aggregation. No ladder checkpoints on
  disk: evaluate in-process; store final checkpoint only (~0.5 GB × ~40 runs ≈ 20 GB).
