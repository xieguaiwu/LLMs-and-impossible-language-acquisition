#!/usr/bin/env python3
"""cald_value_ppl.py — offline V-token perplexity for the CALD arms (§10c-13 A2/G2_cald).

WHY
    The CALD conditions (cald_local / cald_long / cald_shuf) embed K→V filler-gap
    dependencies inside bigram-generated filler text. Training ppl over the whole
    sentence is dominated by fillers; the registered primary metric (prereg
    §10c-13 A2, G2_cald/CALD1) is the NLL of the V tokens **given the full
    prefix**, computed offline on final/ weights. CALD1 = [ln ppl_V(shuf) −
    ln ppl_V(long)]_GPT2 ≥ 0.5 nats; CALD3 uses distance-bucket spread ≤ 0.3,
    so this script also reports V-NLL bucketed by the K→V distance tertiles.

Protocol fidelity
    Sentences are token-ID lines (``cald_affected.test``), evaluated with the
    same Kallini convention as ``train_exp1.get_perplexities``: position 0 is
    never predicted (labels shifted), padding uses EOS with an attention mask,
    per-sentence ppl is the geometric mean over predicted tokens. The V set is
    ``manifest["kv_ids"][200:400]`` (K = first 200, V = last 200; make_cald_conditions.py).

Selftest (--selftest)
    No CALD data exists yet at registration time, so --selftest runs the whole
    pipeline on 5 synthetic CALD-shaped sentences with a tiny randomly-seeded
    GPT-2 (no weights loaded, CPU): checks V-position discovery, K→V distances
    (local variant d=1), bucket construction and that V-ppl is a real
    perplexity (>1, finite). Deterministic seed 0.

USAGE
    python3 cald_value_ppl.py --model-dir results_cald/babylm_cald_long_100M/seed0/final \
        --condition cald_long --out cald_v_ppl_long_s0.json
    python3 cald_value_ppl.py --selftest
    # code-path check on any token-ID file (no cald data yet):
    python3 cald_value_ppl.py --model-dir <final_dir> --condition cald_long \
        --test-file <dir>/babylm_test_affected/not_random_affected.test \
        --v-ids 1892,3673,329,13,14 --limit 20
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments_v2" / "kallini_repro"))

import train_exp1 as G                          # noqa: E402  (protocol source: EOS pad, masking)

K_OFFSET = 200          # manifest kv_ids[:200] = K tokens, kv_ids[200:400] = V tokens
DEFAULT_BATCH = 8


def resolve_data(args) -> tuple[list[list[int]], list[int], list[int], dict]:
    """Locate the token-ID test file + manifest, return (sentences, K_ids, V_ids, meta).

    Overrides (--test-file / --manifest / --v-ids) exist so the code path can be
    validated before the CALD generator has produced data (audit note: overrides
    are for smoke/debug only — adjudication runs must use the frozen manifest).
    """
    data_root = G.BABYLM_DATA_PATH / "babylm_data_perturbed" / f"babylm_{args.condition}"
    test_file = Path(args.test_file) if args.test_file else \
        data_root / "babylm_test_affected" / "cald_affected.test"
    man_file = Path(args.manifest) if args.manifest else data_root / "_cald_manifest.json"
    if not test_file.is_file():
        raise FileNotFoundError(
            f"{test_file} not found — CALD data not generated yet? "
            f"Use --selftest for the weight-free code-path check, or --test-file/--manifest overrides.")
    lines = [l for l in test_file.read_text().splitlines() if l.strip()]
    sents = [[int(t) for t in l.split()] for l in lines]
    if args.v_ids:
        v_ids = [int(x) for x in args.v_ids.split(",") if x.strip()]
        k_ids: list[int] = []
        meta = {"source": str(test_file), "manifest": None,
                "v_ids_override": v_ids, "k_ids_override": True}
    else:
        if not man_file.is_file():
            raise FileNotFoundError(f"{man_file} not found — pass --manifest or --v-ids")
        man = json.loads(man_file.read_text())
        kv = [int(x) for x in man["kv_ids"]]
        assert len(kv) >= 2 * K_OFFSET, f"manifest kv_ids too short: {len(kv)}"
        k_ids, v_ids = kv[:K_OFFSET], kv[K_OFFSET:2 * K_OFFSET]
        meta = {"source": str(test_file), "manifest": str(man_file),
                "condition": man.get("condition"), "variant": man.get("variant"),
                "K_pairs": man.get("K"), "d_range": man.get("d_range"),
                "len_range": man.get("len_range")}
    if args.limit:
        sents = sents[: args.limit]
    return sents, k_ids, v_ids, meta


def _per_position_nll(model, token_lists: list[list[int]], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-position NLL (nats) for all predicted positions, Kallini convention.

    Returns (nll, pred_mask): both (B, L); position i is predicted iff i >= 1
    and the token is real (attention mask 1). Mirrors get_perplexities masking
    but keeps positions un-aggregated so V tokens can be selected afterwards.
    """
    input_ids = G.create_input_ids(token_lists, G.EOS_TOKEN_ID).to(device)
    attention_mask = G.create_attention_mask(token_lists).to(device)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    logp = torch.log_softmax(logits, dim=-1)
    nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)     # (B, L-1) for positions 1..L-1
    pred_mask = attention_mask[:, 1:].to(nll.dtype)               # never predict pads
    full = torch.zeros(input_ids.shape, dtype=nll.dtype, device=nll.device)
    full[:, 1:] = nll
    fmask = torch.zeros(input_ids.shape, dtype=pred_mask.dtype, device=pred_mask.device)
    fmask[:, 1:] = pred_mask
    return full, fmask


@torch.no_grad()
def evaluate(model, sents: list[list[int]], k_ids: list[int], v_ids: list[int],
             device: str, batch: int = DEFAULT_BATCH) -> dict:
    """Score all sentences; aggregate V-token NLL, sentence gmean and distance buckets."""
    K, V = set(k_ids), set(v_ids)
    per_sent: list[dict] = []
    v_nll_all: list[float] = []
    dist_all: list[int] = []
    dist_nll: list[float] = []
    sent_gmean_nll: list[float] = []
    n_v_no_k = 0
    for i in range(0, len(sents), batch):
        chunk = [s[: G.SEQ_LEN] for s in sents[i : i + batch] if len(s) >= 2]
        if not chunk:
            continue
        nll, mask = _per_position_nll(model, chunk, device)
        for b, toks in enumerate(chunk):
            m = mask[b].bool()
            positions = torch.nonzero(m).flatten().tolist()
            tok_pos = [p for p in positions if toks[p] in V]
            k_pos = [p for p in positions if toks[p] in K]
            v_nlls, dists = [], []
            for p in tok_pos:
                prev_k = [q for q in k_pos if q < p]
                v_nlls.append(float(nll[b, p]))
                if prev_k:
                    d = p - max(prev_k)
                    dists.append(d)
                    dist_all.append(d)
                    dist_nll.append(float(nll[b, p]))
                else:
                    n_v_no_k += 1
            nlls_all = [float(nll[b, p]) for p in positions]
            v_nll_all.extend(v_nlls)
            per_sent.append({
                "sent_idx": i + b, "n_tokens": len(toks),
                "v_positions": tok_pos, "v_nll": [round(x, 5) for x in v_nlls],
                "k_v_distance": dists,
                "sent_mean_nll": round(float(np.mean(nlls_all)), 5) if nlls_all else None,
            })
            if nlls_all:
                sent_gmean_nll.append(float(np.mean(nlls_all)))

    def gmean(vals: list[float]) -> float | None:
        return math.exp(float(np.mean(vals))) if vals else None

    v_ppl = gmean(v_nll_all)
    # distance tertiles over the OBSERVED K→V distances (deterministic; disclosed)
    buckets: dict[str, dict] = {}
    if dist_all:
        edges = np.percentile(np.asarray(dist_all, dtype=float), [100 / 3, 200 / 3])
        names = ("t1_low", "t2_mid", "t3_high")
        bounds = [(-1, edges[0]), (edges[0], edges[1]), (edges[1], float("inf"))]
        for name, (lo, hi) in zip(names, bounds):
            sel = [n for n, d in zip(dist_nll, dist_all) if lo < d <= hi]
            buckets[name] = {
                "n": len(sel),
                "mean_v_nll": round(float(np.mean(sel)), 5) if sel else None,
                "dist_range": [int(lo) + 1 if lo > -1 else None,
                               int(hi) if math.isfinite(hi) else None],
            }
        spread = (max(b["mean_v_nll"] for b in buckets.values() if b["mean_v_nll"] is not None)
                  - min(b["mean_v_nll"] for b in buckets.values() if b["mean_v_nll"] is not None))
    else:
        edges, spread = None, None
    summary = {
        "n_sents": len(per_sent),
        "n_v_tokens": len(v_nll_all),
        "n_v_tokens_without_k": n_v_no_k,
        "v_nll_mean": round(float(np.mean(v_nll_all)), 5) if v_nll_all else None,
        "v_ppl": round(v_ppl, 4) if v_ppl else None,
        "v_nll_median": round(float(np.median(v_nll_all)), 5) if v_nll_all else None,
        "sent_gmean_ppl": round(gmean(sent_gmean_nll), 4) if sent_gmean_nll else None,
        "distance_tertile_edges": [round(float(e), 2) for e in edges] if edges is not None else None,
        "distance_buckets": buckets,
        "bucket_spread_nats": round(float(spread), 5) if spread is not None else None,
        "distance_mean": round(float(np.mean(dist_all)), 3) if dist_all else None,
    }
    return {"summary": summary, "per_sentence": per_sent}


def selftest() -> int:
    """Synthetic CALD-shaped sentences + tiny random GPT-2; full pipeline, no weights."""
    torch.manual_seed(0)
    from transformers import GPT2Config, GPT2LMHeadModel
    cfg = GPT2Config(vocab_size=50257, n_positions=64, n_embd=32, n_layer=2, n_head=4,
                     bos_token_id=G.EOS_TOKEN_ID, eos_token_id=G.EOS_TOKEN_ID)
    model = GPT2LMHeadModel(cfg).eval()
    kv_ids = list(range(30000, 30400))
    k_ids, v_ids = kv_ids[:200], kv_ids[200:400]
    rng = np.random.default_rng(0)
    fillers = list(range(1, 51))
    sents: list[list[int]] = []
    expect_v = 0
    for j in range(5):
        variant = "local" if j % 2 == 0 else "long"
        pre = list(rng.choice(fillers, size=int(rng.integers(2, 5))))
        post = list(rng.choice(fillers, size=int(rng.integers(2, 5))))
        f = list(rng.choice(fillers, size=int(rng.integers(8, 32))))
        k = int(k_ids[rng.integers(0, 200)])
        v = int(v_ids[rng.integers(0, 200)])
        s = pre + ([k, v] + f if variant == "local" else [k] + f + [v]) + post
        sents.append(s)
        expect_v += 1
    res = evaluate(model, sents, k_ids, v_ids, device="cpu", batch=2)
    s = res["summary"]
    ok = True
    ok &= s["n_sents"] == 5 and s["n_v_tokens"] == expect_v
    ok &= s["v_ppl"] is not None and math.isfinite(s["v_ppl"]) and s["v_ppl"] > 1.0
    ok &= s["sent_gmean_ppl"] is not None and s["sent_gmean_ppl"] > 1.0
    ok &= set(s["distance_buckets"]) == {"t1_low", "t2_mid", "t3_high"}
    ok &= all(b["n"] > 0 for b in s["distance_buckets"].values())
    # local variant must contribute d=1 distances; long variant d>=8 (generator ranges)
    ds = [d for r in res["per_sentence"] for d in r["k_v_distance"]]
    ok &= 1 in ds and max(ds) >= 8
    ok &= s["n_v_tokens_without_k"] == 0
    print("selftest:", "PASS" if ok else "FAIL")
    print(json.dumps(s, indent=2))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=None, help="HF dir (results_cald/<cell>/final)")
    ap.add_argument("--condition", default=None, help="e.g. cald_long / cald_local / cald_shuf")
    ap.add_argument("--out", default=None, help="output JSON path")
    ap.add_argument("--test-file", default=None, help="override: any token-ID test file (smoke)")
    ap.add_argument("--manifest", default=None, help="override: _cald_manifest.json path")
    ap.add_argument("--v-ids", default=None, help="override: comma-separated V token ids (smoke)")
    ap.add_argument("--limit", type=int, default=0, help="first N sentences only (smoke)")
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--selftest", action="store_true",
                    help="synthetic sentences + tiny random GPT-2, no weights (code-path check)")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.model_dir or not args.condition:
        ap.error("--model-dir and --condition are required (or use --selftest)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained(args.model_dir).eval().to(device)
    sents, k_ids, v_ids, meta = resolve_data(args)
    res = evaluate(model, sents, k_ids, v_ids, device=device, batch=args.batch)
    report = {
        "model_dir": args.model_dir, "condition": args.condition, "device": device,
        "metric": "V-token NLL given full prefix (Kallini position-0 convention)",
        "v_token_source": "manifest kv_ids[200:400]" if not args.v_ids else "--v-ids override",
        "data_meta": meta, "limit": args.limit, **res,
    }
    out = Path(args.out) if args.out else \
        Path(args.model_dir).parent / f"cald_value_ppl_{args.condition}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
