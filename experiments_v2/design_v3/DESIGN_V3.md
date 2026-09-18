# DESIGN_V3 — Final Preregistered Design (synthesis of three independent design agents)

**Status: FROZEN 2026-09-19T01:3x+08:00, before v3 data collection.** This document is the
single source of truth for the v3 revision. Source documents (all agent-authored, kept as
evidence): `EXPDESIGN_V3.md` (prometheus, design), `REDTEAM.md` (momus, adversarial review),
`STATS_PLAN_V3.md` (ultrabrain, statistics), `KALLINI_DESIGN_NOTES.md` (Kallini audit + v2 signals).
Any deviation → append to preregistration.md §10 with timestamp.

## A. Frozen decisions (adopted verbatim from the agents)

1. **Primary corpus = BabyLM 100M** via the kallini_repro pipeline (their perturb.py for
   S/R classes; `design_v3/v3_conditions.py` for class P). Toy SVO = frozen appendix arm
   (H8 artifact diagnosis + the toy 3× overfit that motivated H7-on-BabyLM).
2. **Estimand rule (REDTEAM blocker #2):** all cross-condition inference is on
   **within-family marker-matched deltas**, never raw natural-vs-markered orderings.
   Primary contrast: `parity_word − fixed_start` (family F1, adjudicated first).
   Content-token-only scoring (marker positions masked) reported as robustness.
3. **Erratum-first presentation (REDTEAM blocker #1):** the revision opens with the H8
   polluted-vs-clean side-by-side; the retired per-step t-test figures and the
   "loss+ppl more objective than Kallini" claim are dropped (ppl = e^loss).
4. **Budget semantics (REDTEAM blocker #3):** budgets defined in tokens/epochs
   (3000×128×1024 ≈ 0.39B ≈ 3 epochs base; H7 2× = 6000 steps ≈ 6 epochs; 3× = 9000 ≈
   Kallini's 11-epoch token budget). The batch-4/seq-128 arm is labeled "toy diagnostics".
5. **Replication claims live only in kallini_repro** (their perturb.py verbatim). Our
   `bare_reverse` is labeled "Reverse-bare (no marker)" and is not a replication cell (REDTEAM #5).
6. **LSTM arm fixes (REDTEAM #4):** pad-masked loss (implemented), per-family LR check on
   natural condition only (3 LR × 2 seeds, cheap, on CPU), epochs-matched budget, wording
   "no detectable difference at this budget (CIs reported)" — equivalence (TOST d=0.8)
   only with the n=17 extension; at n=5 unpaired TOST power = 0.000 (sim), n=17 unpaired = 0.47,
   n≈17 paired = 0.87–1.00 → the extension must be paired-seed.
7. **Test-set hygiene (REDTEAM #7):** exact-duplicate-filter the 10k eval draw (report the
   near-dup rate); curves reported whole (no best-checkpoint selection); ladder is
   {100,300,500,1000,2000,3000} for base, scaled for extended budgets; {100,300} excluded
   from AUC (warmup-dominated).
8. **Probe suite (REDTEAM #8 fixes):** P3 length extrapolation built on BabyLM with
   asymmetric caps (train ≤ 60 words, eval 60–200) — the SVO P3 was empty by construction;
   P4 domain-dissociation probe adjudicates word- vs BPE-parity within-model on the
   disagreement subset; P1/P2 report branch-matched deltas (obeying-first vs violating-last
   AND obeying-last vs violating-first); SVO probe results voided.

## B. Hypotheses (from STATS_PLAN_V3 §6, verbatim wording in that file)

- **H9** (replication, no α spent): T0 Kallini panel ordering, criterion-based (τ_a ≥ 0.75).
- **H10** (directional, paired, one-sided, adjudicated FIRST): parity_word > fixed_start.
- **H11**: parity_tok deficit ≠ parity_word deficit; P4 identifies the learned domain.
- **H12** (architecture axis): Δ(control − impossible) > 0 for GPT-2 on ≥2 classes;
  Δ_LSTM reported per §B.6 wording rules.
- **H7′/H8** post-hoc with first-signal disclosure blocks; H7 overfit guard:
  test-channel +0.1 nats at extended budget ⇒ "not testable" on this corpus.

## C. Holm families (9–11 confirmatory rows; else BH q=0.10 exploratory bucket)

F1={parity_word vs fixed_start}; F2={parity_tok vs fixed_start, vs fixed_end};
F3={negtok vs parity_word}; F4={architecture deltas on shuffle_control / full_reverse /
parity_word}; F5={H7 within-condition, blocked at n=1}. Primary metric `ppl_gmean_final`
on ln-ppl; paired-by-seed wherever seed lists match; Shapiro gate is a continuity check,
not evidence (n=5 power 9–24%).

## D. Dispatch (all automatic; single-3080 + cpu2)

| Phase | Runs | Est. | Status |
|---|---|---|---|
| v2 SVO toy (GPU): replication 40 + ext 20 + polluted 15 + probes | 75 | ~4 h | running |
| v2 BabyLM replication (GPU): 3 cond × 5 seeds, batch-4 protocol | 15 | ~1.5 h | queued in unit |
| kallini_repro S/R panel (GPU): 9 langs × 3 seeds | 27 | ~4 d | chained unit, RUN_V3=1 armed |
| **v3 P-class grid (GPU): parity_word, fixed_start, parity_tok, negtok × 3 seeds + H7 2× ×3** | **14** | **57.6 h** | same unit, queued |
| probes P1–P4 (GPU inference) | — | ~3 h | chained |
| LSTM v3 arm (cpu2, after tiny/lstm SVO queue): 7 conds × 5 seeds | 35 | ~17.5 h | to chain |
| optional extensions (P3 fixed_end n=5, H7 3×, seeds→5, LSTM n=17) | — | queued tiers | — |

Run-matrix per-run protocol: kallini_repro trainer verbatim (batch 128, seq 1024, 3000
steps, warmup 300→6e-4 linear decay, stability flags, per-sentence ppl geometric mean at
ladder) — the H7 3× upgrade doubles as Kallini-token-budget fidelity.

## E. Claim-to-design mapping

See EXPDESIGN_V3.md §6 (9-row table). The philosophical section (Ryle/Piaget/Halliday)
moves to a clearly labeled commentary section decoupled from evidence (REDTEAM #9),
with an explicit UG-as-architecture paragraph and the outcome-to-interpretation matrix.

## F. Known failure modes — v2 incident rules (all pre-empted in code)

result-dir key = args.dataset verbatim; skip-if-done honors budget suffixes; special
tokens registered + resized (unit-tested); per-host result branches; Holm families frozen
ex ante; ≤13 planned inferential rows (13-row budget from STATS_PLAN_V3 §10).
