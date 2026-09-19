# EXPDESIGN_V3 — Experimental Design for the Paper Revision

**Paper:** *LLMs and Impossible Language Acquisition* (Ziyan Wang, v2→v3 revision)
**Status:** DESIGN v1.0 (2026-09-19). Frozen before v3 data collection; any deviation must be
logged in §7 and in the preregistration deviation log (experiments_v2/preregistration.md §10).
**Inputs:** KALLINI_DESIGN_NOTES.md (Kallini et al. 2024 design audit), preregistration.md (v2),
kallini_repro/README.md (running reproduction), current arXiv tex (hypotheses + theory claims).
**Hardware:** 1× RTX 3080 10GB (GPU arm) + 1× 4-core CPU box (LSTM arm).

---

## 0. Design summary (read this first)

v3 **switches the primary corpus from the 10k synthetic SVO toy corpus to BabyLM 100M** and
adopts Kallini's five design principles (per-class controls, impossibility continuum,
held-out perturbed-test geometric-mean ppl at a checkpoint ladder, marker entropy controls,
filter-enforced cross-condition sentence-set identity, 5 seeds). The toy SVO corpus is
**demoted to a frozen appendix/sanity arm** (its v2 results already exist; H7 and parity_tok
were both shown to be unanswerable on it: test-channel overfitting at 3× budget; word-parity ≡
BPE-parity because every template word is a single BPE token).

The paper's own contributions are preserved and strengthened:

- **(a) Architecture axis:** GPT-2-small (124M) vs capacity-matched LSTM (~39M) on *byte-identical*
  languages and an identical eval protocol. LSTM runs on the CPU box (Kallini has no LSTM arm —
  this remains the paper's core incremental result).
- **(b) Parity-negation counting-rule family** with marker controls
  (`fixed_start`/`fixed_end`/`negtok`) and a **BPE-token-parity variant (`parity_tok`) that is
  only meaningful on BabyLM** (word parity ≠ BPE parity on natural text; on the toy corpus they
  were degenerately identical).
- **(c) Behavioral probes** for the parity rule: minimal pairs, length extrapolation, hidden-state
  diagnostic, plus a new **domain-dissociation probe (P4)** that adjudicates word- vs token-parity
  directly on the sentences where the two domains disagree.
- **(d) Budget-dependence arm (H7) on BabyLM**, where the test channel is not saturated
  (the toy corpus overfit at 3×: train 0.55 but test 1.92→2.79).

Everything is scheduled so that the **new GPT-2 grid fits in ≤ ~2.5 GPU-days (57.6 h)**, with
the Kallini *Shuffle/*Reverse replication credited to the already-queued `kallini_repro`
workstream (27 runs, ~97 h, separate budget line, same seeds and protocol → mergeable).

---

## 1. Condition grid

### 1.1 Class structure and controls

Kallini's central principle: **every language class carries its own control**, and conditions
within a class differ only in the impossibility of the transformation. Four classes:

| Class | Language | Transformation | Control of class | Marker | Vocab | Seeds (base) | Tier |
|---|---|---|---|---|---|---|---|
| **S** (\*Shuffle, Kallini replication) | `shuffle_control` | none (English) | — (control of S) | none | 50257 | 0,14,41 | **T0 reuse** |
| S | `shuffle_nondet` | per-sentence random shuffle | shuffle_control | none | 50257 | 0,14,41 | T0 reuse |
| S | `shuffle_deterministic` | length-bucketed deterministic shuffle | shuffle_control | none | 50257 | 0,14,41 | T0 reuse |
| S | `shuffle_local3` / `shuffle_local10` | window-3 / window-10 local shuffle | shuffle_control | none | 50257 | 0,14,41 | T0 reuse |
| S | `shuffle_evenodd` | even positions ≫ odd positions | shuffle_control | none | 50257 | 0,14,41 | T0 reuse |
| **R** (\*Reverse, Kallini replication) | `reverse_control` | none + marker `R` at random position | — (control of R) | `R` (special) | 50258 | 0,14,41 | T0 reuse |
| R | `partial_reverse` | reverse after marker | reverse_control | `R` | 50258 | 0,14,41 | T0 reuse |
| R | `full_reverse` | reverse whole sentence | reverse_control | `R` | 50258 | 0,14,41 | T0 reuse |
| **P** (parity-negation counting family — **ours**) | `parity_word` | "Not" at **end** if odd **word** count, at **start** if even (paper's rule) | `fixed_start` (primary), `natural` (no-marker anchor) | `Not` (natural word) | 50257 | 0,14,41 | **P0** |
| P | `parity_tok` | same rule over **BPE-token** count (GPT-2 tokenization of the raw sentence, pre-marker) | `fixed_start` | `Not` | 50257 | 0,14,41 | **P1** |
| P | `negtok` | word parity, marker = reserved `<NEG>` token | `parity_word` (marker-identity contrast) | `<NEG>` (special) | 50258 | 0,14,41 | **P1** |
| P | `fixed_start` | "Not" always sentence-initial, no counting rule | — (**control of P**) | `Not` | 50257 | 0,14,41 | **P0** |
| P | `fixed_end` | "Not" always sentence-final | `fixed_start` (position control) | `Not` | 50257 | 0,14 | **P3 ext** |
| **B** (budget arm — **ours**, H7) | `natural_3x` | = shuffle_control, 2×/3× steps | its own 1× run (within-condition contrast) | none | 50257 | 0 | **P2** |
| B | `parity_word_3x` | = parity_word, 2×/3× steps | its own 1× run | `Not` | 50257 | 0 | **P2** |

Control logic for class P (this is the design's answer to prereg confound **C1**):

- **Primary control = `fixed_start`**: identical marker token ("Not", capitalized,
  sentence-initial surface form), identical marker frequency, **no counting rule**. The
  paper's central inference (rule learnability) rests on `parity_word` vs `fixed_start`.
- **`fixed_start` + `fixed_end` jointly** bracket the parity condition's bimodal position
  distribution (~50% start / ~50% end), matching the marker's marginal position entropy —
  the same entropy-control move Kallini makes with `R` in the \*Reverse class.
- **`negtok`** isolates marker *identity* (reserved special token vs natural word) from the
  rule; `negtok` vs `parity_word` is a pure marker contrast at fixed rule.
- **`natural` (= `shuffle_control`)** is the no-marker anchor shared with classes S/R; it is
  *not* the primary control for P (marker confound), exactly as `NoReverse` carries a marker.

### 1.2 Toy SVO appendix/sanity arm (frozen)

No new compute. Reuse the v2 toy-corpus cells (30 cpu2 cells + 40 GPU v2-replication cells)
and the clean-regeneration fix. Role in v3: (i) appendix sanity arm showing the pipeline
replicates the *original paper's* numbers only under the polluted corpus (H8, `Original:`
duplication artifact); (ii) demonstration that `parity_tok ≡ parity_word` on the toy corpus
(single-BPE template words), which is the *motivation* for the BabyLM parity_tok arm;
(iii) the toy 3× overfit (train 0.55, test 1.92→2.79) that motivates running H7 on BabyLM.

### 1.3 Dataset construction rules (all classes)

1. **Source.** BabyLM 100M via HF mirror `Sree1994/babylm_100M`, single concatenated genre
   file; sentence segmentation via the repro's `shim_tag.py` regex shim (Kallini's Stanza
   pipeline is only needed for \*Hop, which stays out of scope — see §7).
2. **Global filters (Kallini principle: cross-condition sentence-set identity).** Sentences
   of 2–200 whitespace words. Any sentence that *any* grid condition cannot transform
   (e.g., marker placement undefined, < 2 words) is dropped from **all** conditions. All
   conditions therefore train and test on the identical sentence-ID set.
3. **Parity domains (confound C2 made explicit).**
   - *Word domain:* whitespace token count of the raw sentence.
   - *Token domain:* GPT-2 BPE token count of the raw sentence **before marker insertion**.
   - Report the **disagreement rate** (sentences where word parity ≠ BPE parity); expected
     ~30–50% on natural text (vs 0% on the toy corpus). The P4 probe (§4) conditions on
     exactly this subset.
4. **Markers.** `Not` is a natural vocabulary word (no embedding resize; matches the paper's
   original rule). `<NEG>` and `R` are registered special tokens with
   `add_special_tokens` + `resize_token_embeddings` (**the v2 negtok bug — unregistered
   marker BPE-fragmented into `< NEG >` — must never recur; add a unit test**).
5. **Deduplication.** Train split kept verbatim (Kallini fidelity; BabyLM duplication is part
   of the replicated phenomenon) but **duplication statistics are reported** (H8 lesson).
   **Test set is exact-duplicate-filtered** before the 10,000-sentence draw.
6. **Split.** Train / held-out with the repro's split; test = 10,000 held-out sentences
   (same sentence IDs across all conditions), perturbed per condition at eval time.

---

## 2. Training protocol

### 2.1 GPT-2 arm (RTX 3080 10GB) — BabyLM

Identical to the verified kallini_repro trainer (which in turn follows Kallini App. B):

| Hyperparameter | Value |
|---|---|
| Model | HF `gpt2` (124M), **from scratch** |
| Sequence / packing | seq 1024; numpy rng(seed) shuffle of sentence lines → EOS-join → 1024-token chunks (Kallini verbatim) |
| Effective batch | **128** (grad accumulation; Kallini used 512 — see §7) |
| Optimizer steps | **3000** (base grid); **6000** (H7 2×) ; 9000 (3× upgrade, queued) |
| LR | 6e-4, linear warmup 300, **linear decay to 0** at final step (repro's documented assumption) |
| Stability flags | `reorder_and_upcast_attn=True`, `scale_attn_by_inverse_layer_idx=True` |
| Seeds | [0, 14, 41] base — **shared with kallini_repro → paired cross-class comparisons at n=3**; extension [53, 96] brings headline P-class cells to 5 |
| Checkpoint saving | final checkpoint retained **only** for probe-target conditions (`parity_word` ×3, `fixed_start` ×3); all other runs evaluate in-process and discard weights |

Token-budget accounting: 3000 steps × 128 × 1024 ≈ 0.39B tokens ≈ 3 epochs of BabyLM 100M
(Kallini: 1.57B ≈ 11 epochs). The H7 arm at 3× (9000 steps ≈ 1.18B ≈ 9 epochs) therefore
doubles as a **Kallini-token-budget fidelity arm** — one run answers two questions.

### 2.2 LSTM arm (4-core CPU box) — BabyLM, identical languages

| Hyperparameter | Value |
|---|---|
| Model | `lstm_matched` 640d / 2-layer, ~39M params (v2 definition). Capacity ledger reported in the paper: 39M LSTM vs GPT-2-small 124M total (≈85M non-embedding) — the tight matched pair remains `gpt2_tiny`(44M) vs `lstm_matched`(39M) from the toy appendix; the 124M-vs-39M contrast is the paper's "architecture axis at scale", not a capacity-controlled comparison, and must be worded as such |
| Data | **Byte-identical perturbed datasets and packing as the GPT-2 arm** (same sentence IDs, same filters, same test set) |
| Batch / steps | batch 32, seq 512, steps scaled to ≈ GPT-2 token budget as CPU time allows; ~30 min/run (given) |
| Optimizer | v2 LSTM regime (Adam, per `training/train_lm.py`), frozen and logged per run |
| Seeds | [0, 14, 41, 53, 96] — full 5 seeds, cheap on CPU |
| Conditions | 7: `shuffle_control`, `full_reverse`, `reverse_control`, `parity_word`, `parity_tok`, `negtok`, `fixed_start` |
| Eval | same perturbed-test geometric-mean ppl at the same ladder |

CPU cost: 7 × 5 = 35 runs × 0.5 h ≈ 17.5 h (≈ 6–8 h wall with 2–3 parallel workers).

---

## 3. Eval protocol

1. **Metric (Kallini verbatim).** Per-sentence perplexity on the 10,000-sentence
   condition-perturbed held-out test set; **geometric mean** per checkpoint. Train loss is
   logged for process inspection only and is **never** used for inference (v2 lesson: the
   toy 3× arm showed train/test dissociation).
2. **Checkpoint ladder.** {100, 300, 500, 1000, 2000, 3000} (repro subset of Kallini's
   100-step ladder), evaluated in-process; same ladder for LSTM (scaled proportionally to
   its step count). H7 runs ladder at {300, 1000, 2000, 4000, 5000, 6000} (2×) /
   {..., 9000} (3×).
3. **Endpoints.**
   - *Primary:* geomean test ppl at the final checkpoint (3000 base / 6000 H7).
   - *Secondary:* AUC of geomean ppl over the log-step ladder (learning-efficiency claim);
     argmin-checkpoint ppl (asymptotic learnability claim).
4. **Statistics (inherits prereg §5, adapted).** Unit of analysis = one scalar per seed.
   Welch's t (normality pass) else Mann-Whitney; Holm-Bonferroni within family; Cohen's d
   with bootstrap 95% CI. **Paired-by-seed comparisons** wherever seeds are shared
   (all GPU base cells share seeds with T0). TOST equivalence (bound d = 0.8) only for LSTM
   null claims, and only at n ≥ 17 seeds (cheap on CPU: +24 runs ≈ 12 h) — otherwise a null
   is reported as "no detectable difference at n=5", never as equivalence.
5. **Continuum analysis.** Class S/R yields the Kallini Figure-2 ordering as a *replication
   panel*; class P is inserted into the same figure as a new class with its control, giving
   the paper a single unified impossibility-continuum figure on BabyLM.

---

## 4. Probe plan (no additional training; ~2–3 h GPU inference)

Run on `parity_word` seeds {0,14,41} final checkpoints; `fixed_start` checkpoints serve as
negative control (marker present, no rule → probes should be at chance).

| Probe | Construction | Metric | Prediction if the rule was learned |
|---|---|---|---|
| **P1 Minimal pairs** | 500 held-out pairs differing only in marker position: rule-obeying vs rule-violating | marker surprisal delta Δ = S(violating) − S(obeying); Wilcoxon over pairs, per seed | Δ > 0; %pairs(Δ>0) well above 50 |
| **P2 Length extrapolation** | 200 pairs longer than any training sentence (length > 99th pct of train; up to the 200-word filter cap) | same Δ | Δ > 0 persists (rule, not memorization); memorizing model → Δ ≈ 0 |
| **P3 Hidden-state diagnostic** | logistic regression (sklearn) on last-layer hidden state at the marker position → binary parity class; 5-fold CV | accuracy vs 50% chance | accuracy ≫ 50% on parity_word; ≈ 50% on fixed_start |
| **P4 Domain dissociation (new, BabyLM-only)** | on the word↔BPE parity **disagreement subset** (§1.3.3), pairs where the marker placement obeys word parity vs BPE parity | marker surprisal: which domain's rule-violations hurt more, in the `parity_word` model vs the `parity_tok` model | `parity_word` model tracks word parity, `parity_tok` model tracks BPE parity → adjudicates C2 (the paper must state which domain the rule lives in) |

P4 is the decisive probe for the paper's parity claim: it converts the "which parity domain?"
question from a between-condition ppl comparison into a within-model behavioral measurement.

---

## 5. Compute budget

Per-run GPU cost (measured): **3.6 h** at 3000 steps / batch 128 / seq 1024 incl. 6-checkpoint
eval; 7.2 h at 6000; 10.8 h at 9000. Budget cap for the new GPT-2 grid: **≤ ~2.5 GPU-days (60 h)**.

### Priority-ordered run matrix — GPT-2 / BabyLM (new cells)

| Pri | Condition | Seeds | Steps | Runs | Hours | Cum. |
|---|---|---|---|---|---|---|
| **T0** | *Kallini replication (reused, separate budget):* shuffle_control, nondet, deterministic, local3, local10, evenodd, reverse_control, partial, full × 3 seeds | 0,14,41 | 3000 | 27 | 97.2 | (outside cap) |
| **P0** | `parity_word` | 0,14,41 | 3000 | 3 | 10.8 | 10.8 |
| **P0** | `fixed_start` | 0,14,41 | 3000 | 3 | 10.8 | 21.6 |
| **P1** | `parity_tok` | 0,14,41 | 3000 | 3 | 10.8 | 32.4 |
| **P1** | `negtok` | 0,14,41 | 3000 | 3 | 10.8 | 43.2 |
| **P2** | `natural_3x` (H7) | 0 | 6000 | 1 | 7.2 | 50.4 |
| **P2** | `parity_word_3x` (H7) | 0 | 6000 | 1 | 7.2 | **57.6** |
| P3 ext | `fixed_end` | 0,14 | 3000 | 2 | 7.2 | 64.8 |
| P4 ext | H7 3× upgrade (`natural_3x`, `parity_word_3x`) | 0 | 9000 | 2 | 21.6 | 86.4 |
| P5 ext | seed extension [53,96]: parity_word, fixed_start, parity_tok, negtok | 2×4 | 3000 | 8 | 28.8 | 115.2 |
| P6 ext | probe inference | — | — | — | ~3 | — |

- **Base grid = P0+P1+P2 = 14 runs, 57.6 h ≈ 2.4 GPU-days ✓ under the 2.5-day cap**
  (2.4 h margin for data prep, eval overruns, and reruns of the known failure modes:
  dataset-key collisions, skip-if-done suffix bugs, unregistered special tokens — all three
  bit us in v2).
- `fixed_end` at n=2 is acceptable for base because the toy data showed fixed_start ≡ fixed_end
  exactly (1.677 / 1.677); its full seeds sit in P5.
- H7 base runs at **2× (6000 steps)**: the toy signal emerged between 1× and 3×; 2× is the
  first informative point on BabyLM. The 3× upgrade (P4) doubles as the Kallini-token-budget
  fidelity run and is the **first** extension priority.
- If any P0/P1 run is lost, rerun within the margin before touching extension tiers.

### LSTM arm (CPU box, separate from GPU cap)

35 runs × 0.5 h = 17.5 h (+12 h optional n=17 equivalence extension for the LSTM null on
`shuffle_control` vs `parity_word`), ≈ 6–8 h wall with parallel workers.

### Totals

| Arm | New compute |
|---|---|
| GPU base grid (in-cap) | 57.6 h ≈ **2.4 GPU-days** |
| GPU extension queue (optional, priority-ordered) | 57.6 h |
| GPU T0 kallini_repro (already queued/running) | 97.2 h |
| CPU LSTM | ~17.5–29.5 h |
| Probes | ~3 h GPU inference |

---

## 6. Claim-to-design mapping

| # | Paper claim (current tex → v3) | Design element that carries it | Hypothesis |
|---|---|---|---|
| 1 | Chomsky: "LLMs cannot distinguish possible from impossible languages" — empirically false | T0 continuum replication (geomean ppl ordering over 9 Kallini languages) + class P inserted into the same continuum | H9: ordering replicates; ppl increases with impossibility |
| 2 | Transformer architecture harbors structure-favoring inductive bias | Architecture axis: GPT-2 vs LSTM on byte-identical languages/protocol (§2.2), per-class gap Δ = ppl(control) − ppl(impossible) compared across architectures | H12: Δ_GPT2 > 0 for ≥ 2 classes; Δ_LSTM ≈ 0 (reported as no-difference or equivalence per §3.4 power) |
| 3 | The parity-negation deficit reflects the counting rule, not the marker (C1) | `parity_word` vs `fixed_start` (marker-matched) vs `natural` (no-marker anchor); `fixed_end` position control; `negtok` identity control | H10: parity_word > fixed_start ⇒ rule effect; parity_word ≈ fixed_start ⇒ paper's parity interpretation must be revised (adjudicate **before** other comparisons, per prereg guardrail) |
| 4 | The rule's domain must be stated (C2) | `parity_tok` on BabyLM (where word ≠ BPE parity, unlike toy) + P4 dissociation probe | H11: parity_tok deficit ≠ parity_word deficit; P4 identifies the learned domain |
| 5 | Models learn the rule, not surface marker distribution | Probes P1–P3 with fixed_start negative control | H6 (prereg): Δ>0 minimal pairs; Δ>0 on extrapolated lengths; probe accuracy ≫ chance |
| 6 | The natural-vs-impossible gap is budget-dependent (H7; toy arm showed test-channel overfit, so BabyLM is the correct venue) | Budget arm B: 1× vs 2× (base) vs 3× (= Kallini token budget, upgrade) on natural & parity_word | H7′: gap(natural − parity_word) grows with budget on **test** ppl; if train↓ but test↑ → overfitting regime, report as such |
| 7 | Original toy-corpus magnitudes were a duplication artifact (H8) | Frozen toy appendix arm + train-duplication statistics (§1.3.5); test dedup | H8 (already evidenced; appendix) |
| 8 | "Both loss and perplexity" methodological point vs Kallini | Train loss logged as process evidence; all inference on held-out test geomean ppl at the ladder | — (methods claim, absorbed into §3) |
| 9 | Philosophical claim (functionalist/empiricist paradigm shift) | Not carried by v3 runs; strengthened rhetorically by #1–#6 (a graded, controlled, multi-seed continuum is precisely the "empirical natural science" methodology the paper argues for) | — |

Guardrails carried over from the prereg: H10 is adjudicated first; a replication failure is
reported as such; equivalence language only with the n=17 LSTM extension.

---

## 7. Deviations-from-Kallini rationale

| # | Deviation | Rationale |
|---|---|---|
| 1 | Effective batch 512 → **128** (steps 3000 held) | Single 3080 10GB VRAM. Token budget 0.39B ≈ 3 epochs vs their 1.57B ≈ 11. Mitigated by the H7 arm (3× ≈ 9 epochs ≈ their budget); all cross-condition comparisons are within our own budget, so the grid is internally valid |
| 2 | LR: linear decay to 0 after warmup 300→6e-4 | Paper specifies only warmup; their 4000-warmup note implies decay. Same choice as the verified repro → comparability with T0 |
| 3 | Checkpoint ladder subset {100,300,500,1000,2000,3000} of their 100-step ladder | In-process eval (no disk checkpoints) on one GPU; 6 points resolve the learning-trajectory claim |
| 4 | \*Hop class excluded | Stanza POS/morph GPU cost; their Exp-1 ppl differences within \*Hop were minimal. **Partially compensated** by our parity class: it is a counting-rule family (the linguistically interesting property of \*Hop) *with* marker controls, and `parity_tok` directly probes token- vs word-counting — arguably a cleaner counting-rule test than TokenHop/WordHop |
| 5 | HF mirror single-genre file + regex shim segmentation (not their 10-genre Stanza pipeline) | Verbatim from the repro; Shuffle/Reverse/parity perturbations consume no POS annotations |
| 6 | Class P ("Not"/`<NEG>` markers) is new, not in Kallini | It is the paper's contribution. Control construction follows their marker-entropy principle: `fixed_start`/`fixed_end` match marker identity, capitalization and marginal position distribution; `negtok` isolates marker identity. Note "Not" is a natural frequent word — this is why the **primary** control is marker-matched `fixed_start`, not bare `natural` |
| 7 | Test set exact-duplicate-filtered; train duplication kept + reported | Test dedup protects the measurement channel (H8 lesson); train kept verbatim for Kallini comparability, with duplication statistics disclosed |
| 8 | Base 3 seeds [0,14,41] (Kallini: 5) → extension tier to 5 | GPU cap 2.5 days. Seeds chosen = Kallini's first three → all T0/P-class cells pairable per-seed; P5 upgrades headline cells to 5 seeds |
| 9 | LSTM arm and budget arm added | Not in Kallini; they are the paper's contributions (a) and (d). LSTM hyperparams are per-architecture (documented), languages/protocol identical |
| 10 | H7 at 2× in base, 3× as first extension | Toy-corpus 3× overfit made test-channel saturation the risk; BabyLM at 2× = 6 epochs is the first unsaturated informative point; 3× simultaneously serves as the Kallini-token-budget fidelity run |

**Known failure modes to pre-empt (v2 incident log):** result-directory key must be
`args.dataset` verbatim (no aliases); skip-if-done checks must include `_3x`/`_ext` suffixes;
special tokens (`<NEG>`, `R`) must be registered + embeddings resized, with a unit test;
per-host result branches to avoid force-push clobbering.

---

## 2.2a LSTM arm — cpu2 budget deviations (registered 2026-09-19)

Registered by the cpu2 runner session; the LSTM arm is **budget-limited** and must be
reported only in the budget-dependent wording of the F8/H7 family.

| Item | Design (§2.2) | As run on cpu2 | Why |
|---|---|---|---|
| Steps | "scaled to ≈ GPT-2 token budget as CPU time allows" | **300** | 4-core CPU throughput |
| Seq len | 512 | **256** | batch 32 × seq 512 × 50257-vocab logits = 3.3 GB fwd + 3.3 GB bwd → OOM-killed (exit 137, measured) |
| Batch | 32 | **32 effective** (micro 4 × accum 8) | identical effective batch, bounded peak memory |
| Eval sample | same 10k test sentences | **first 2000 of that same 10k sample** (nested subset, not a new draw) | one full 10k eval ≈ 30 min on CPU |
| Workers | "2–3 parallel workers" | **2 workers × 2 torch threads** | 4 cores, 7.7 GB RAM (2 workers × 512 MB mmap'd packed stream) |

**Token budget**: 300 × 32 × 256 = **2.46e6** tokens vs the GPT-2 arm's
3000 × 128 × 1024 = **3.93e8** → **ratio ≈ 1/160**.

**Binding wording constraint**: this arm licenses **no** equal-budget architecture-axis
claim. Only: "no detectable difference **at this budget**" (F8/H7 family).
Epochs-matched (3 epochs ≈ 3.9e8 tokens) would need **~160 h per cell** on cpu2:
the GPT-2 arm measures ~27k tok/s (13.1e6 tokens / 8.1 min) while `lstm_matched`
measures **~680 tok/s** here (12 s/step at 8192 tokens/step) — the 50257 × 640 vocab
head, not the recurrence, is the bottleneck.

**Byte-identity**: `packing_equivalence_check.py` proves the cpu2 numpy packer
reproduces `train_exp1.load_packed_dataset`'s token stream exactly (3 seeds, multi-file,
token-for-token) — the LSTM arm imports `train_exp1` as the single protocol source.
