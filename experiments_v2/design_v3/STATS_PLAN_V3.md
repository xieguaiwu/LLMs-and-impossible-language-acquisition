# STATS_PLAN_V3 — Statistical Analysis Plan for the v3 BabyLM Grid

**Status:** FROZEN 2026-09-19 (written before v3 data collection; deviations must be logged in
`preregistration.md` §10 and in §8 below). Implements EXPDESIGN_V3 §3.3–3.4 at full precision.
**Analysis code:** `analysis/stats_tests_v3.py` (extends `analysis/stats_tests.py`: reuses
`holm_bonferroni`, `cohens_d`, `d_bootstrap_ci`, `tost_welch` verbatim; adds `tost_paired`,
family router, paired-by-seed wiring).

---

## 1. Unit of analysis and checkpoint-curve summaries

**Unit of analysis = ONE SCALAR PER SEED** per (architecture × condition × budget_tag) cell.
Per-step and per-checkpoint values are **never** independent observations (the v2
pseudo-replication lesson; serial correlation AR(1) > 0.99).

**Primary observation.** For run r = (arch, condition, budget_tag, seed): at each checkpoint
s on the eval ladder, per-sentence perplexity on the 10,000-sentence condition-perturbed
held-out test set (identical sentence IDs across conditions), geometric mean computed with
the verbatim `get_perplexities` from `kallini_repro`:

    ppl_gmean(r, s) = exp( (1/N) Σ_i ln ppl_i ),  N = 10,000

**Curve summaries (exactly three, fixed ex ante):**

| Column | Definition | Role |
|---|---|---|
| `ppl_gmean_final` | ppl_gmean(r, T); T = final checkpoint (3000 base / 6000 H7-2× / 9000 H7-3×) | **PRIMARY endpoint for all families** |
| `auc_logstep` | trapezoidal mean of Y(r,s)=ln ppl_gmean over ln(s), s ∈ [s_min, T]; s_min = 500 (base) / 1000 (H7) = first post-warmup ladder point (warmup = 300); the {100, 300} points are warmup-dominated and excluded from AUC (kept for figures) | secondary (learning-efficiency claims) |
| `ppl_argmin`, `argmin_step` | min and argmin over the ladder | tertiary (asymptotic-learnability descriptor) |

All contrasts are computed on **ln(ppl) = per-sentence cross-entropy (nats)**; ppl values are
reported as `exp(Δ)` ratios for interpretability.

**Curve-shape statistics (descriptive; no per-step p-values, ever):**
- `first_gap_step`: smallest s where **all** seeds show Y_cond(s) > Y_ctrl(s) (seed-unanimity
  descriptor, valid at any n).
- CS1 sup-t band on the per-seed gap curve g(s) = Y_cond(s) − Y_ctrl(s): seed-level bootstrap
  (B = 10,000), sup-t simultaneous band; "gap interval" = band excludes 0 over ≥ 2 consecutive
  checkpoints. At n = 3 report the min–max envelope only (no inferential wording).
- CS2: Spearman(Y, ln s) per condition (learning-curve monotonicity descriptor).

---

## 2. Test selection flow (Shapiro/Levene gate at n ≤ 5)

Per comparison, in order (frozen, matches `stats_tests.test_cell`):

1. **Shapiro-Wilk per group** (α = .05; skipped if n < 3).
2. **Levene** (scipy default = Brown–Forsythe, center='median').
3. **Welch's t** (two-sided by default; one-sided where the hypothesis is directional) when
   both groups pass Shapiro; otherwise **Mann-Whitney U** (`method='exact'` for n ≤ 8).
4. Holm-Bonferroni within family (§3). 5. Cohen's d + bootstrap CI (§6). 6. TOST (§5).

**Honesty caveats (pre-registered, from simulation, 200k reps, 2026-09-19):**

- Shapiro at n=5 is nearly uninformative: rejection rate 0.047 under N(0,1) but only 0.10 /
  0.24 / 0.09 / 0.16 under lognormal(0.5), lognormal(1.0), t(3), exponential. The gate is
  retained **for continuity with the v2 frozen code, not as evidence of normality**; Shapiro
  p-values are reported as descriptive columns only.
- Exact-test granularity at tiny n: Wilcoxon/sign-flip min two-sided p = 2/2⁵ = **0.0625 at
  n=5** and 2/8 = **0.25 at n=3** (never < .05); MWU exact min p = 2/C(10,5) = 0.0079 at 5v5
  but **0.10 at 3v3** (never < .05). Hence at n=3 the **only** test that can reach p < .05 is
  parametric (paired t, df=2) — every n=3 row is flagged `parametric-dependent` and headline
  claims require the seed extension to n=5.

**Paired-by-seed design (primary wherever seed lists match).** All GPT-2 base cells share
seeds [0, 14, 41] (also shared with the T0 kallini_repro runs); LSTM cells use
[0, 14, 41, 53, 96]. Seed governs init + packing rng, so cross-condition contrasts are
seed-paired. *Pairing-validity check:* Spearman ρ̂ of `ppl_gmean_final` between the two cells
across seeds; if ρ̂ ≤ 0, that comparison is demoted to unpaired (Welch) for that row.
Paired rows: primary test = paired t (df = n−1), sensitivity = exact sign-flip/Wilcoxon
(granularity caps disclosed above). Unpaired rows: Welch primary, exact MWU sensitivity.

---

## 3. Holm-Bonferroni families (v3 grid, precisely)

One primary summary (`ppl_gmean_final`) per family; `auc_logstep` rows are exploratory.
Scale anchor for effect sizes and margins: σ̂_cell = pooled within-cell SD across seeds of the
two cells compared (on ln-ppl). Holm step-down within each family, α = .05, full-precision
adjusted p (v2 lesson: rounding can break p_holm ≥ p_raw monotonicity).

| Family | Members (exact contrasts, arch, seeds, metric, test) | m |
|---|---|---|
| **F1** | `parity_word` vs `fixed_start` — gpt2, seeds [0,14,41]→[0..96], paired, ppl_gmean_final, paired-t one-sided (H10: rule effect) | 1 |
| **F2** | `parity_tok` vs `fixed_start`; `parity_tok` vs `fixed_end` — gpt2, paired, ppl_gmean_final, paired-t one-sided (H11: token-parity deficit exists vs marker-only controls). `fixed_end` enters only once it has ≥ 3 seeds (base n=2 → defer, log in §8) | 2 |
| **F3** | `negtok` vs `parity_word` — gpt2, paired, ppl_gmean_final, paired-t one-sided (H4′: deficit(negtok) ≥ deficit(parity_word), i.e. Y_parity_word ≤ Y_negtok; marker-identity contrast at fixed rule) | 1 |
| **F4** | Architecture delta contrasts, seed-paired within arch on shared seeds [0,14,41]: **F4a** Y_shuffle_control^GPT2 − Y_shuffle_control^LSTM (two-sided: do architectures differ on natural English); **F4b** [Y_full_reverse − Y_shuffle_control]^GPT2 − [same]^LSTM > 0 (one-sided); **F4c** [Y_parity_word − Y_shuffle_control]^GPT2 − [same]^LSTM > 0 (one-sided) | 3 |
| **F5** | H7′ within-condition budget contrasts: Y_natural(2×) vs Y_natural(1×); Y_parity_word(2×) vs Y_parity_word(1×); + the 3× analogs when the 3× upgrade runs (m → 4). **Blocked at base tier (seed 0 only, n=1): descriptive reporting, no p-values; activate at n ≥ 3** | 2 (+2) |

**Budget:** 9 confirmatory rows now, ≤ 11 with the 3× upgrade. Plus ≤ 3 pre-named TOST rows
(§5) and the H9 replication criterion (criterion-based, spends no α). Total ≤ 13 planned
inferential rows. Anything else → exploratory bucket (Benjamini-Hochberg q = 0.10) + §8 entry.

**Guardrails:** F1 (H10) is adjudicated **before** any other comparison is interpreted.
A replication failure is reported as such; no metric shopping (one primary summary per family).

---

## 4. Not covered by the families (explicitly out of inferential scope)

- The 9 Kallini T0 languages' Figure-2 ordering (H9) is adjudicated by a **replication
  criterion, not a p-test**: Kendall τ_a between pre-registered impossibility ranks and
  per-condition seed-mean Y_final over the 9 languages, threshold τ_a ≥ 0.75 with the
  seed-bootstrap 95% CI excluding 0 and ≥ 7/8 pairwise orderings concordant with Kallini's
  published ordering. Pre-registered ranks: shuffle_control 0; reverse_control 1; fixed_start
  1; negtok 2; partial_reverse 2; full_reverse 3; parity_word 3; parity_tok 4;
  shuffle_evenodd 4; shuffle_local3 5; shuffle_local10 6; shuffle_deterministic 7;
  shuffle_nondet 8 (ties allowed, τ-b).
- Probes P1–P4 (incl. the decisive P4 domain-dissociation probe on the word↔BPE disagreement
  subset): per-seed Wilcoxon over 500 pairs, seed-level sign-consistency criterion, fixed_start
  as negative control; reported per EXPDESIGN §4, descriptive, no α spent.
- H8 (§7): frozen toy appendix, no new tests.

---

## 5. Equivalence testing (TOST) — LSTM null claims only

**Rule (inherited from prereg §5.5, made precise):** TOST against margin
**Δ = 0.8 · σ̂_cell** (the conventional "large" anchor) is admissible **only for the LSTM
no-natural-language-bias claims**, pre-named:

- **TOST-L1:** full_reverse vs shuffle_control, LSTM, paired by seed
- **TOST-L2:** parity_word vs shuffle_control, LSTM, paired by seed
- **TOST-L3:** fixed_start vs shuffle_control, LSTM (marker-cost check; exploratory TOST)

Implementation (`tost_paired`, df = n−1): t₁ = (d̄+Δ)/(s_d/√n), t₂ = (d̄−Δ)/(s_d/√n),
p = max(P(T>t₁), P(T<t₂)); equivalent iff p < .05 (both one-sided tests reject).
Unpaired fallback (ρ̂ ≤ 0): Welch-based `tost_welch` (as in v2 code).

**Power math (simulation, 200k reps, α = .05 per one-sided test, margin 0.8σ̂_cell,
true Δ = 0):**

| Design | n=3 | n=5 | n=10 | n=17 | n=30 |
|---|---|---|---|---|---|
| Paired, σ_d = 1.0·σ̂_cell | 0.07 | 0.14 | 0.52 | **0.87** | — |
| Paired, σ_d = 0.5·σ̂_cell (ρ̂ ≈ .88) | 0.37 | 0.80 | 1.00 | ≈1.00 | — |
| Unpaired (two-sample) | 0.00 | **0.00** | 0.04 | 0.47 | **0.84** |

- **n≈17/group gives 80%+ power for the seed-matched (paired) LSTM cells** — this is the
  operative plan (LSTM extension: +24 runs ≈ 12 h CPU, per EXPDESIGN §5).
- **Unpaired TOST at n=5 is structurally impossible (power 0.000)**; even n=17 unpaired is
  only 0.47 (would need n≈30). Unpaired rows therefore never yield equivalence at these N.

**Verdict vocabulary (binding):**

| Seeds/cell | Allowed equivalence wording |
|---|---|
| n = 5 | *"no detectable difference at n=5"* — **NEVER "equivalent" / "no bias"** |
| n = 17 paired (LSTM cells) | *"statistically equivalent at d = 0.8 (TOST, p_holm < .05)"* — paper-level null claim |
| n = 17 unpaired | downgraded to "no detectable difference (TOST power 0.47, insufficient)" |

A stricter margin (d = 0.5) may be reported additionally; at n=17 it requires measured
pairing ρ̂ ≥ 0.8 (power 0.98 at σ_d=0.5 vs 0.28 at σ_d=1.0).

---

## 6. Effect sizes with bootstrap CIs

- **d column** = standardized effect on the ln-ppl scale: **dz** = mean(diffs)/SD(diffs) for
  paired rows; pooled **d** (Hedges-J corrected, reported as `g_j`) for unpaired rows.
- **95% CI: percentile bootstrap, 10,000 resamples over seeds** (resampling seeds within each
  cell; paired rows resample seed indices once, jointly). Fixed rng seed 42 for reproducibility.
- Also reported: raw Δ in nats with the exact t-CI on the log scale, and ppl ratio exp(Δ) with
  its CI (primary interpretive object when a cell's variance is near zero — bootstrap d-CIs
  degenerate there; flagged `degenerate_d` in those rows).

---

## 6b. Multiple-comparison budget (summary)

Confirmatory (Holm, α=.05): F1 = 1, F2 = 2, F3 = 1, F4 = 3, F5 = 2 (+2 at 3×) → **9–11 rows**.
Acceptance tests: TOST-L1/L2 (+L3 exploratory) = 2–3 rows. Criterion-based: H9 = 1 row.
Exploratory bucket (BH q=.10): everything else, incl. `auc_logstep` twins, CS1 bands,
parity_word-vs-parity_tok direct contrast, P-probe seed summaries. Every row — significant or
not — is published in `stats_tests_v3.csv`; no selective reporting.

---

## 7. Pre-registration wording (directional) + post-hoc diagnostics

**H9 (replication, criterion-based).** *On BabyLM, GPT-2-small geomean test ppl increases with
pre-registered impossibility rank across the 9 Kallini languages: τ_a ≥ 0.75 (seed-bootstrap
95% CI excluding 0) with ≥ 7/8 pairwise orderings concordant with Kallini Fig. 2 (esp.
shuffle_control < shuffle_nondet; reverse_* ≈ reverse_control). Falsification: τ_a < 0.75 or
the control not lowest.* Seeds [0,14,41]; n=3 per cell.

**H10 (rule vs marker; adjudicated FIRST).** *If the parity-negation deficit reflects the
counting rule, then Y_parity_word > Y_fixed_start (F1, paired, one-sided). If instead
Y_parity_word ≈ Y_fixed_start, the paper's parity interpretation must be revised to a
marker-distribution effect. Equivalence side of this adjudication uses the §5 wording rules
(no equivalence language at n<17).*

**H11 (parity domain).** *Token-count parity is a learnable rule but costlier than
marker-only conditions: Y_parity_tok > Y_fixed_start and Y_parity_tok > Y_fixed_end (F2,
paired, one-sided). Which domain the rule lives in is adjudicated by probe P4 on the word↔BPE
disagreement subset, not by the ppl contrast alone; the direct parity_tok-vs-parity_word
contrast is reported descriptively.*

**H12 (architecture axis).** *The impossibility gap is architecture-dependent:
Δ_c^GPT2 − Δ_c^LSTM > 0 for c ∈ {full_reverse, parity_word} (F4b, F4c; paired by seed on
shared seeds). If F4b rejects but F4c does not, scope the architecture claim to
global-reordering languages. LSTM nulls are settled by TOST (§5), never by "p > .05".*

**H7′ (post-hoc, diagnostic, non-confirmatory — first-signal disclosure).** *Registered
2026-09-19 AFTER the v2 toy-corpus extended-budget arm showed train 1.71→0.55 with test ppl
1.92→2.79 (test-channel overfitting on the toy corpus) — disclosed here and in prereg §10/§11.
On BabyLM (where the test channel is not saturated), the natural-vs-impossible gap grows with
budget: [Y_natural − Y_parity_word] at 2× (6000 steps) exceeds the same gap at 1× (3000);
3× (9000) analog when run (F5, one-sided, paired by seed). Overfit guard: if
Y_natural(2×/3×) > Y_natural(1×) + 0.1 nats (test-channel degradation), H7′ is adjudicated
"not testable — overfitting regime" and no emergence claim is made (the toy-arm outcome).*
**Inference blocked at base tier (n=1 seed): descriptive curves only until n ≥ 3.**

**H8 (post-hoc, diagnostic, non-confirmatory — first-signal disclosure).** *Registered
2026-09-18 (v2 prereg §11) after the clean-corpus GPU early signal showed natural ≈ reversed
(test PPL 1.918 vs 1.915), contradicting the paper's Exp-1 direction, and the polluted arm
showed natural_polluted train 1.71→1.36. v3 status: frozen toy-corpus appendix evidence +
BabyLM train-duplication statistics (reported, §1.3.5 of EXPDESIGN_V3) + test-set dedup as
the mitigation. No new confirmatory tests; licensed output is a *correction/qualification*
of the original Exp-1 magnitudes, not a confirmatory claim. A BabyLM polluted-corpus arm
would be a new exploratory diagnostic and requires a §8 entry before running.*

---

## 8. Deviation log (v3)

| Date | Deviation | Reason |
|---|---|---|
| 2026-09-19 | TOST power clause of prereg §5.5 refined: n≈17 is powered **only for seed-paired** TOST (0.87); unpaired equivalence needs n≈30; n=5 TOST power 0.00 | 200k-rep simulation (§5) — prevents an unfalsifiable equivalence claim |
| 2026-09-19 | n=3 rows flagged `parametric-dependent`; exact rank tests structurally incapable at n=3 (min p 0.10–0.25) | exact granularity math (§2) |
| 2026-09-19 | Shapiro/Levene gate retained for v2-code continuity, not as normality evidence | Shapiro power 0.09–0.24 under severe non-normality at n=5 (§2) |

*(append further entries here, timestamped, before any data peek beyond what §7 discloses)*

---

## 8b. Results-table template (single tidy table, `aggregated/stats_tests_v3.csv`)

One row per comparison; all rows published (significant or not). mean±SD on the ln-ppl (nats)
scale; paired rows use paired tests (test column carries the design); `verdict` vocabulary:
`diff>0_holm.05 | diff<0_holm.05 | diff≠0_holm.05 | no_detectable_diff_n{k} |
equivalent_d0.8 | not_equivalent | inconclusive | blocked_n<3 | criterion_pass | criterion_fail`.

```csv
dataset,model,metric,cond1,cond2,n1,n2,mean1,sd1,mean2,sd2,test,p_raw,p_holm,d,d_ci_low,d_ci_high,tost_p,verdict
babylm,gpt2,ppl_gmean_final,parity_word,fixed_start,3,3,2.314,0.041,2.102,0.038,paired_t,0.012,0.012,1.85,0.42,3.05,NA,diff>0_holm.05
babylm,gpt2,ppl_gmean_final,parity_tok,fixed_start,3,3,2.556,0.051,2.102,0.038,paired_t,0.004,0.008,2.41,0.71,3.88,NA,diff>0_holm.05
babylm,lstm_matched,ppl_gmean_final,parity_word,shuffle_control,17,17,1.902,0.043,1.887,0.041,paired_t,0.318,0.318,0.31,-0.22,0.84,0.041,equivalent_d0.8
babylm,gpt2,ppl_gmean_final,natural_3x_vs_1x,natural,1,1,1.642,NA,1.598,NA,blocked,NA,NA,NA,NA,NA,NA,blocked_n<3
```
*(rows 1–4 are format placeholders, not results; `mean1/sd1` etc. are y-scale nats;
`d` = dz for paired rows, Hedges-g for unpaired; `d_ci_*` from 10k seed bootstrap.)*

Companion inputs (fixed schema): `aggregated/checkpoint_evals_v3.csv`
(`run_id, arch, condition, budget_tag, seed, step, tokens_seen, n_test_sents, ppl_gmean,
ppl_gmean_boot_lo, ppl_gmean_boot_hi`) → `aggregated/per_seed_v3.csv`
(`run_id, arch, condition, budget_tag, seed, ppl_gmean_final, auc_logstep, ppl_argmin,
argmin_step, first_gap_step, tokens_seen_final, git_commit`). Aggregation decomposes ops-level
keys (`natural_3x`) into `condition=natural, budget_tag=3x` — unit test required (v2 alias bug).
