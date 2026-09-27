#!/usr/bin/env python3
"""probes_ddg.py — DDG 终点探针（§10c-14）：demonstration-dependence gap。

三通道逐句配对（同一评估句集）：
  zero-shot : 评估句单独评测（= 现有 ppl 协议口径，位置 0 不预测）
  K-shot    : 评估句前拼 K=5 条**同条件 train split** 示范（固定 seed=0 采样、固定顺序）
  K-nat     : 同 K=5 示范但取自 shuffle_control train split（自然示范对照）

主判据量 DDG*(c) = DDG_raw(same) − DDG_raw(natural)，DDG_raw = mean lnNLL_zero − mean lnNLL_K
（只对评估句 token 计 NLL——示范前缀做条件不计损失；Kallini 约定：位置 0 永不预测）。
确定性截断：示范块 + 评估句 > 1024 token ⇒ 依序换更短示范句（写死规则，非随机）。

评估句集 = 该条件 eval 池 10k 抽样（numpy rng(seed)）的**前 2000 句**（嵌套子集，F8 口径）。
数据源：token-ID 行（train/*.train 示范、test_affected/*.test 评估），与 trainer 同布局。

用法（Phase A exploratory，n=500；终判 n=2000）:
  python3 probes_ddg.py --model-dir <final/> --condition parity_word --n 500 --out ddg_parity_word.json
  python3 probes_ddg.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("KALLINI_DATA_PATH", "/root/kallini_data")) / "babylm_data_perturbed"
K = 5
CTX_LIMIT = 1024
NATURAL_COND = "shuffle_control"


def condition_files(cond: str) -> tuple[list[Path], list[Path]]:
    d = DATA / f"babylm_{cond}"
    tr = sorted((d / "babylm_100M").glob("*.train"))
    te = sorted((d / "babylm_test_affected").glob("*_affected.test"))
    assert tr, f"no train files for {cond}"
    assert te, f"no test files for {cond}"
    return tr, te


def load_train_lines(cond: str, cap: int = 200000) -> list[list[int]]:
    """train split token-ID 句（示范池）；cap 行防内存。"""
    out: list[list[int]] = []
    for f in condition_files(cond)[0]:
        for line in f.read_text().splitlines():
            toks = [int(t) for t in line.split()]
            if toks:
                out.append(toks)
            if len(out) >= cap:
                return out
    return out


def load_eval_nested(cond: str, seed: int, n_draw: int = 10000) -> list[list[int]]:
    """trainer 口径：test 池全读 → rng(seed).permutation → 前 n_draw；本探针取前 2000（嵌套）。"""
    lines: list[str] = []
    for f in condition_files(cond)[1]:
        lines.extend(l for l in f.read_text().splitlines() if l.strip())
    pool_n = len(lines)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(lines))[:n_draw]
    ev = [[int(t) for t in lines[i].split()] for i in idx]
    return ev


def pick_demos(pool: list[list[int]], seed: int, k: int, budget: int, eval_len: int) -> list[list[int]]:
    """固定 seed=0 采样起点、固定顺序取 k 条；超预算依序换更短示范（确定性规则）。"""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pool))
    chosen: list[list[int]] = []
    for i in order:
        cand = pool[i]
        trial = chosen + [cand]
        if sum(len(s) + 1 for s in trial) + eval_len <= budget:
            chosen = trial
        else:
            # 超预算：换更短示范——从剩余里找能塞进去的最短者（依序扫描，确定性）
            rest = [pool[j] for j in order if pool[j] not in chosen]
            replaced = False
            for j in rest:
                for pos in range(len(chosen)):
                    if sum(len(s) + 1 for s in (chosen[:pos] + [j] + chosen[pos + 1:])) + eval_len <= budget:
                        chosen[pos] = j
                        replaced = True
                        break
                if replaced:
                    break
            if not replaced:
                break
        if len(chosen) >= k:
            break
    return chosen[:k]


@torch.no_grad()
def nll_channel(model, eval_sents: list[list[int]], demos: list[list[int]] | None,
                device: str) -> dict:
    """逐句 NLL（只计评估句 token；示范前缀作条件）。demos=None ⇒ zero-shot。"""
    per_sent: list[float] = []
    lens: list[int] = []
    for ev in eval_sents:
        if demos:
            seq = []
            for d in demos:
                seq.extend(d)
                seq.append(EOS)
            start = len(seq)
            seq.extend(ev)
        else:
            seq = list(ev)
            start = 0
        x = torch.tensor([seq], dtype=torch.long, device=device)
        out = model(input_ids=x)
        logp = torch.log_softmax(out.logits[0, :-1, :].float(), dim=-1)
        labels = x[0, 1:]
        nll = -logp[torch.arange(labels.numel()), labels]
        # 只计评估句 token 的 NLL（demo 段与位置 0 除外）
        ev_slice = nll[max(start - 1, 0):]
        per_sent.append(float(ev_slice.mean()))
        lens.append(len(ev))
    return {"per_sentence": per_sent, "n": len(per_sent),
            "mean_lnNLL": float(np.mean(per_sent)), "mean_len": float(np.mean(lens))}


EOS = 50256


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--condition", default=None)
    ap.add_argument("--n", type=int, default=2000, help="评估句数（嵌套 2000；exploratory 可 500）")
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        pass
    elif not (args.model_dir and args.condition):
        ap.error("--model-dir 与 --condition 必填（--selftest 除外）")
    device = "cpu"

    if args.selftest:
        # 合成数据自检：通道机制 + 截断规则 + 配对相减
        ev = [[100 + i for i in range(30)], [200 + i for i in range(40)]]
        pool = [[300 + i for i in range(10)], [400 + i for i in range(12)], [500 + i for i in range(11)], [600 + i for i in range(13)]]
        d = pick_demos(pool, 0, 3, 1024, 30)
        assert len(d) == 3
        assert sum(len(s) + 1 for s in d) + 30 <= 1024
        print("SELFTEST_OK (demo selection + budget rule)")
        return 0

    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained(args.model_dir).eval().to(device)

    ev_all = load_eval_nested(args.condition, args.seed)
    ev = ev_all[: args.n]
    tr_pool = load_train_lines(args.condition)
    nat_pool = load_train_lines(NATURAL_COND)
    # 示范选择：按句逐条选（预算依赖 eval_len）——统一用第一条 eval 长度近似选一套，
    # 超预算句在 nll_channel 内不再换（截断规则在选示范时已按最长评估句保守执行）
    max_eval = max(len(s) for s in ev)
    demos_same = pick_demos(tr_pool, args.seed, args.k, CTX_LIMIT, max_eval)
    demos_nat = pick_demos(nat_pool, args.seed, args.k, CTX_LIMIT, max_eval)

    res = {
        "condition": args.condition, "model_dir": args.model_dir, "K": args.k,
        "n": len(ev), "seed": args.seed, "exploratory": args.n < 2000,
        "demo_ids_same": [len(d) for d in demos_same],
        "demo_ids_nat": [len(d) for d in demos_nat],
        "zero": nll_channel(model, ev, None, device),
        "kshot_same": nll_channel(model, ev, demos_same, device),
        "kshot_nat": nll_channel(model, ev, demos_nat, device),
    }
    d_raw_same = res["zero"]["mean_lnNLL"] - res["kshot_same"]["mean_lnNLL"]
    d_raw_nat = res["zero"]["mean_lnNLL"] - res["kshot_nat"]["mean_lnNLL"]
    res["DDG_raw_same"] = round(d_raw_same, 5)
    res["DDG_raw_nat"] = round(d_raw_nat, 5)
    res["DDG_star"] = round(d_raw_same - d_raw_nat, 5)
    out = Path(args.out) if args.out else Path(args.model_dir).parent / f"ddg_{args.condition}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("zero", "kshot_same", "kshot_nat")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
