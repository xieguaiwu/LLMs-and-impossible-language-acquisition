"""CPU smoke tests for experiments_v2 (no GPU / no transformers required).

Run:  cd <repo root> && python3 -m pytest experiments_v2/tests/test_all.py -q
  or: python3 experiments_v2/tests/test_all.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments_v2" / "data_v2"))
sys.path.insert(0, str(REPO / "experiments_v2" / "training"))
sys.path.insert(0, str(REPO / "experiments_v2" / "analysis"))
sys.path.insert(0, str(REPO / "experiments_v2" / "probes"))

import conditions as C  # noqa: E402
from generate_svo import generate_svo_sentences  # noqa: E402
from metrics import summarize_series, write_run_json  # noqa: E402


# ---------------------------------------------------------------- data ----

def test_generate_svo_deterministic_and_clean():
    a = generate_svo_sentences(200, seed=42)
    b = generate_svo_sentences(200, seed=42)
    assert a == b
    assert all("Original:" not in s for s in a), "v2 corpus must not contain the Original: bug"
    assert all(s.endswith(".") for s in a)
    assert all(3 <= len(s.split()) for s in a)


def test_reverse_invariant():
    sents = generate_svo_sentences(50, seed=7)
    for s in sents:
        r = C.rule_reverse(s)
        assert r.endswith(".")
        body = r[:-1].split()
        orig = C._strip_sentence(s).split()
        assert body == orig[::-1]


def test_parity_rule_matches_word_count():
    sents = generate_svo_sentences(200, seed=3)
    for s in sents:
        p = C.rule_parity_negation(s)
        words = C._strip_sentence(s).split()
        marker_first = p.startswith("Not ")
        marker_last = p.endswith(" Not.")
        assert marker_first != marker_last, f"exactly one marker: {p}"
        if len(words) % 2 == 0:
            assert marker_first and not marker_last
        else:
            assert marker_last and not marker_first


def test_parity_token_unit_differs_when_bpe_unavailable():
    # without transformers the token-unit path must raise a clear error
    try:
        import transformers  # noqa: F401

        has_tf = True
    except ImportError:
        has_tf = False
    if has_tf:
        return  # full check happens on the GPU box
    try:
        C.rule_parity_negation("The cat can see the bird.", unit="token")
        raised = False
    except RuntimeError:
        raised = True
    assert raised


def test_fixed_position_controls():
    s = "The man can like the book."
    assert C.rule_fixed_start_negation(s) == "Not The man can like the book."
    assert C.rule_fixed_end_negation(s) == "The man can like the book Not."


def test_word_shuffle_keeps_lexicon():
    s = "The woman has enjoyed the school."
    sh = C.rule_word_shuffle(s)
    assert sorted(sh[:-1].split()) == sorted(C._strip_sentence(s).split())
    # deterministic
    assert sh == C.rule_word_shuffle(s)


def test_split_indices_disjoint_and_deterministic():
    tr, te = C.split_indices(1000, test_frac=0.05, seed=42)
    assert len(set(tr) & set(te)) == 0
    assert len(tr) + len(te) == 1000
    tr2, te2 = C.split_indices(1000, test_frac=0.05, seed=42)
    assert (tr, te) == (tr2, te2)


def test_apply_condition_all_registered():
    sents = generate_svo_sentences(30, seed=11)
    for name in C.CONDITIONS:
        if name == "parity_negation_tok":
            continue  # requires transformers
        out = C.apply_condition(sents, name)
        assert len(out) == len(sents)
        assert all(isinstance(x, str) and x for x in out)


# ------------------------------------------------------------ metrics ----

def test_summarize_series_known_shape():
    losses = list(np.linspace(2.0, 0.5, 100)) + [0.5] * 20
    s = summarize_series(losses, total_steps=120)
    assert abs(s["final_loss"] - 0.5) < 0.02
    assert s["min_loss"] <= s["final_loss"] + 1e-9
    assert abs(s["auc_loss"] - float(np.mean(losses))) < 1e-6
    assert s["convergence_step"] <= 120
    assert 0 < s["convergence_frac"] <= 1.0
    assert abs(s["final_ppl"] - np.exp(s["final_loss"])) < 1e-4


def test_write_run_json_backward_compatible(tmp_path=None):
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "run.json"
        rec = write_run_json(
            out, run_id="x", experiment="v2_svo", model="gpt2_tiny",
            dataset="svo", condition="natural", seed=42,
            hyperparameters={"lr": 5e-5}, losses=[2.0, 1.5, 1.0],
            test_loss=1.1, total_steps=3,
        )
        loaded = json.loads(out.read_text())
        # old-format consumers read these keys
        for k in ["losses", "final_loss", "total_steps", "dataset"]:
            assert k in loaded
        # new-format consumers read these
        assert loaded["summary"]["test_loss"] == rec["summary"]["test_loss"]
        assert loaded["seed"] == 42


# ------------------------------------------------------------- stats ----

def test_stats_pipeline_recovers_known_effect_and_holm():
    import stats_tests as ST

    rng = np.random.default_rng(0)
    # natural ~N(1.0, 0.05), reversed ~N(2.0, 0.05) over 5 seeds -> huge effect
    rows = []
    for cond, mu in [("natural", 1.0), ("reversed", 2.0), ("parity_negation", 1.4)]:
        for seed, val in enumerate(rng.normal(mu, 0.05, 5), start=42):
            rows.append({"dataset": "svo", "model": "gpt2", "condition": cond,
                         "seed": seed, "final_loss": val})
    df = __import__("pandas").DataFrame(rows)
    out = ST.test_cell(df, "final_loss")
    assert len(out) == 3
    by = {(r["cond1"], r["cond2"]): r for r in out}
    nat_rev = by[("natural", "reversed")]
    assert nat_rev["significant_holm"] is True
    assert nat_rev["cohens_d"] < -5  # d with sd=0.05, diff=1 -> huge negative
    # Holm monotonicity: adjusted >= raw for every comparison
    assert all(r["p_holm"] >= r["p_raw"] for r in out)
    # TOST: huge effect must NOT be declared equivalent
    assert nat_rev["equivalent_at_d0.8"] is False


def test_tost_equivalence_for_null():
    import stats_tests as ST

    rng = np.random.default_rng(1)
    # NOTE: equivalence testing needs power -- at n=5/group a TOST at bound
    # d=0.8 is essentially never passable (needs ~17/group for 80% power).
    # Use n=50 here to demonstrate the acceptance logic itself.
    a = rng.normal(0, 0.1, 50)
    b = rng.normal(0.005, 0.1, 50)  # trivially different, tiny effect
    t = ST.tost_welch(a, b, bound_d=0.8)
    assert t["equivalent_at_d0.8"] is True


def test_holm_bonferroni_monotone_and_bounded():
    import stats_tests as ST

    ps = [0.001, 0.01, 0.04]
    adj = ST.holm_bonferroni(ps)
    assert all(a >= p for a, p in zip(adj, ps))
    assert all(a <= 1.0 for a in adj)
    # order preserved: smallest raw -> smallest adjusted
    assert adj == sorted(adj)


def test_aggregate_collects_v2_format(tmp_path=None):
    import tempfile

    import aggregate_seeds as AG

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for cond in ["natural", "reversed"]:
            for seed in [42, 43]:
                d = root / "svo" / "gpt2" / f"{cond}_seed{seed}"
                d.mkdir(parents=True)
                (d / "training_metrics.json").write_text(json.dumps({
                    "run_id": f"svo_gpt2_{cond}_seed{seed}",
                    "experiment": "v2_svo", "model": "gpt2", "dataset": "svo",
                    "condition": cond, "seed": seed,
                    "summary": {"final_loss": 1.0, "test_ppl": 2.5},
                    "losses": [1.0],
                }))
        # an old-format file must be ignored
        old = root / "legacy.json"
        old.write_text(json.dumps({"losses": [1, 2], "final_loss": 2}))
        df = AG.collect_runs(root)
        assert len(df) == 4
        assert set(df["condition"]) == {"natural", "reversed"}


# ------------------------------------------------------------- probes ----

def test_minimal_pairs_invariants():
    import probes as P

    sents = generate_svo_sentences(60, seed=5)
    pairs = P.make_minimal_pairs(sents, n_pairs=20)
    assert len(pairs) == 20
    for p in pairs:
        ow = p["obeying"].replace("Not ", " ").replace(" Not.", ".").split()
        vw = p["violating"].replace("Not ", " ").replace(" Not.", ".").split()
        assert sorted(ow) == sorted(vw), "minimal pair differs only by marker position"
        assert p["obeying"] != p["violating"]


def test_extrapolation_pairs_exceed_train_lengths():
    import probes as P

    train = generate_svo_sentences(300, seed=9)
    test = generate_svo_sentences(300, seed=13)
    # force some long test sentences
    test = test + ["The man can " + " ".join(["read"] * 6) + " the book."]
    pairs = P.make_extrapolation_pairs(train, test, n_pairs=50)
    max_train = max(len(s.split()) for s in train)
    assert all(p["n_words"] > max_train for p in pairs)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
