# FORMAL_COMPLEXITY.md — a formal ladder for the "impossibility" gradient

> Written 2026-09-20 evening (preregistration §10c-7). **Analysis/writing only: no
> compute, no alpha spent.** Purpose: replace the paper's weakest methodological
> point (its own Limitations admit "we are unable to give a formal definition of
> impossible languages") with a graded, testable formal property, and state the
> predictions the running grid can adjudicate.

## 1. Why this exists

Kallini et al. (2024) and this project both use the word *impossible* for
transformations of English that "violate structure dependence". That is a
linguistic stipulation, not a formal one: reverse, shuffle and counting rules sit
in very different places formally, and the published continuum figure treats them
as one scale. A reviewer can therefore ask what the x-axis *is*.

This document answers that by classifying each condition in the grid by the formal
machinery its transformation requires, and by listing the empirical predictions
that follow. The point is not that formal complexity *is* learnability — it is that
the two can be put into a falsifiable relation on the data we are already
collecting.

## 2. Condition → formal requirement

| Condition (class) | Transformation | Formal requirement on a learner | Formal class of the transform |
|:--|:--|:--|:--|
| `shuffle_control` (S) | identity | none beyond the base language | English (context-free fragment) |
| `shuffle_local3` / `local10` (S) | permute tokens inside a window of k positions | bounded look-back/window memory (k) | strictly **k-local** (⇒ regular) |
| `shuffle_even_odd`, `deterministic21`, `nondeterministic` (S) | global permutation of sentence positions | unbounded permutation memory: you must know the whole sentence to undo it | **non-regular** (needs length-unbounded state) |
| `reverse_control` (R) | original sentence + marker R (no reversal) | marker position only | regular + marker control |
| `reverse_partial`, `reverse_full` (R) | reverse part/all of the sentence | anti-hierarchical reordering; the derivation rules of §"Impossible Language Training" no longer apply | CFL is closed under reversal, so not "impossible" in the CF sense — the violation is **structural/linear**, not formal-grammatical |
| `word_shuffle`, `bare_reverse` (P/R) | shuffle/reverse without markers | as above, minus the marker confound | marker-free analogs |
| `fixed_start`, `fixed_end` (P) | insert marker at a constant position | position-predictable insertion: **no counting** | regular; **entropy/position control** for the parity family |
| `parity_word` (P) | insert marker iff the sentence has an even **word** count | count to 2 over an unbounded length: **MOD-2 counting** | the textbook **non-regular** language (parity is the canonical example) |
| `parity_tok` (P) | same over **BPE token** count | same, and the count must be performed in the token domain the model sees | non-regular; domain-dissociation target (probe P4) |
| `negtok` (P) | as `parity_word`, marker is a reserved `<NEG>` id (vocab +1) | counting + one new symbol | non-regular; marker-identity control |
| `not_random` (P) | marker placed with the **same position distribution** as `parity_word` but **independent of parity** | none (no rule) | regular; **entropy-matched control** (Kallini's NoReverse logic applied to the counting class) |

Two things fall out immediately:

1. **The grid's "impossibility" is not one axis.** It mixes (a) local permutation
   (k-local, regular), (b) global permutation (needs unbounded memory), (c)
   anti-hierarchical reordering (structural violation), and (d) unbounded counting
   (non-regular). A single continuum figure flattens this; the paper should say so
   and use the formal classes as the explanatory ordering instead.
2. **`parity_*` is the only family with a counting requirement**, and it is
   simultaneously the family whose surface form is *easiest* (a marker in 100 % of
   sentences lowers raw perplexity). Raw cross-condition ppl therefore cannot
   identify the counting cost — which is exactly the estimand rule this project
   already froze (REDTEAM #2 / DESIGN_V3 §A.2): use marker-matched within-family
   deltas (`parity_word − fixed_start`, `parity_word − not_random`) plus the
   content-token-only column.

## 3. What transformer theory predicts

* Hahn (2020) shows hard-attention transformers cannot recognise PARITY or Dyck-2;
  the practical reading is that a fixed-depth, finite-precision model has no
  built-in counter, and any counting behaviour must be implemented as a
  finite-precision approximation whose accuracy degrades with length.
* Merrill & Sabharwal and follow-ups refine this: log-precision transformers can
  represent counters up to a length bound (counting to *n* costs log *n* bits), so
  MOD-2 counting over *bounded* inputs is representable — the failure mode is
  **length generalisation**, not representability.
* Net prediction for this grid: if a model solves `parity_word` by *counting*, its
  advantage should survive the P2 length-extrapolation probe (train ≤ 60 words,
  eval 60–200). If it solves the condition by *memorising positional statistics*
  (marker placement distributions), the probe should show a deficit that grows with
  length. P2 in `probes/probes_babylm.py` is exactly this test.

Corollary used by the NoPE arm (§10c-4): a transformer's positional encoding is the
channel through which "linear position" information (the thing Figures 1–2 of the
paper contrast with hierarchy) enters the model. Zeroing it should reduce the
natural-language advantage if that advantage is position-borne.

## 4. Empirical anchor available now: the n-gram floor (preregistration §10c-1)

`analysis/ngram_baseline.py` fits an interpolated absolute-discounting **bigram**
(a lower bound on Kneser-Ney quality) on the same perturbed corpora and scores the
same frozen 10k-sentence draw with the same geometric-mean convention. A bigram
model has **no counter at all**, so it cannot represent any of the counting rules;
its numbers are the pure surface-statistics floor. Measured on cpu2 (6M-token fits,
seeds of the frozen draw = 0):

| condition | ppl (all) | ppl (content-only) | formal requirement |
|:--|--:|--:|:--|
| `shuffle_control` | 480.90 | 480.90 | none |
| `parity_word` | 418.38 | 660.66 | MOD-2 counting |
| `negtok` | 417.81 | 660.66 | MOD-2 counting (+1 symbol) |
| `parity_tok` | 492.86 | 713.22 | MOD-2 counting (token domain) |
| `fixed_start` | 745.89 | 745.89 | none (position control) |
| `reverse_control` | 527.93 | 897.74 | marker control |
| `reverse_full` | 862.93 | 1723.45 | anti-hierarchical reordering |

Three readings, all of which belong in the paper:

1. **The raw-ppl confound is real and is not an LLM artefact.** Even a bigram finds
   the markered conditions *easier* in raw ppl (`parity_word` 418 <
   `shuffle_control` 481), while their content-only ppl is *harder* (661 > 481).
   The marker-entropy warning (REDTEAM #2) is thus confirmed by a model that cannot
   possibly know the counting rule.
2. **The reversal gradient is partly surface.** `reverse_full` (863/1723) is far
   worse than `reverse_control` (528/898) for a bigram too, i.e. some of the
   "impossibility" signal in the R class is recoverable from local statistics.
   Whatever the transformer adds on top of that is the part that needs a
   structure-level explanation.
3. **`parity_word` vs `fixed_start` is not a clean contrast in the bigram either**
   (content 661 vs 746): the two conditions place the marker differently, which
   changes the *content* bigram contexts. This is direct empirical support for the
   registered `not_random` arm (§10c / audit B1): only a marker-position-matched
   control isolates the counting requirement.

## 5. Predictions to report against the grid (pre-specified here, before the
   corresponding cells exist)

| # | Prediction | Data source | Status |
|:--|:--|:--|:--|
| FC1 | Within the P class, the deficit ordering is `fixed_start ≈ fixed_end < negtok ≤ parity_word < parity_tok` | H9 within-class criterion (audit B3) | pending cells |
| FC2 | The content-only deficit of `parity_word` vs `not_random`/`fixed_start` exceeds the bigram's content gap (a counting cost beyond surface statistics) | F1 + n-gram floor | pending cells |
| FC3 | If `parity_word` is learned by counting, P2 (60–200 words) shows no length-dependent collapse; if it is positional memorisation, the deficit grows with length | probes P2 vs ladder probe | pending weights |
| FC4 | Removing positional information (NoPE) reduces the natural-vs-impossible penalty | F7_nope (§10c-4) | pending cells |
| FC5 | The impossibility penalty is larger at 1M/10M tokens than at the full corpus (PoS analog) | F8_datascale (§10c-5) | pending cells |
| FC6 | The penalty shrinks with model scale (355M vs 124M at matched token budget) | F9_model_scale (§10c-6) | pending cells |

Any of these can fail; the point of pre-specifying them here is that the paper can
report the outcome either way without the ordering being chosen after the fact.

## 6. What this document does *not* claim

* It does not claim a formal impossibility result for any condition. CFLs are closed
  under reversal; MOD-2 counting is representable by log-precision transformers up to
  a length bound.
* It does not replace the linguistic argument of the paper (structure dependence);
  it gives that argument a formal axis that can be checked against data.
* The n-gram numbers above are floors from 6M-token fits on **cpu2's copy** of the
  seven available conditions; they will be recomputed on gpu2 with the full pools
  when the grid completes (`analysis/outputs/ngram/`).
