# cpu2 LSTM arm — resource envelope and the 2026-09-20 investigation

> Machine: 4-core CPU box (`cpu2`, 7.7 GB RAM, 3.7 GB swap, no GPU).
> Arm: `lstm_matched` (39M) on the Kallini BabyLM packed streams, `seq 256`, `micro 8`,
> `eff batch 32`, `steps 300`, `eval 2000-of-10000`.
> Recorded by the cpu2 runner session; numbers are measured, not estimated.

---

## 1. Final deployed configuration

```
systemd-run --unit=llm-lstm \
  --property=Restart=on-failure --property=RestartSec=120s --property=StartLimitIntervalSec=0 \
  --property=Nice=15 --property=CPUQuota=200% \
  --property=MemoryHigh=4831838208 --property=MemoryMax=5905580032 \
  --property=MemorySwapMax=1073741824 \
  --setenv=LSTM_WORKERS=1 --setenv=LSTM_THREADS=2 \
  --setenv=LSTM_MICRO_BATCH=8 --setenv=LSTM_EVAL_BATCH=4 --setenv=LSTM_PACK_VERSION=v2 \
  ... /usr/bin/bash experiments_v2/kallini_repro/lstm_v3_queue.sh
```

| Knob | Value | Why |
|---|---|---|
| workers | **1** | each worker needs 2.6–3.2 GB anonymous memory; 2 workers + system exceed 7.7 GB (three global OOM kills on 09-19/09-20) |
| threads | **2** | torch intra-op threads; see §2 — 4 threads is pathological on this box |
| micro batch | 8 (accum 4) | logits 8×256×50257×4 = 412 MB fwd; smaller micro values only add Python-level iteration overhead |
| eval batch | 4 | eval cost is dominated by the 50257-vocab head |
| memory caps | High 4.5 G / Max 5.5 G / swap 1 G | MemoryHigh must sit **above** real usage, or the kernel reclaims continuously (§3) |

Measured on the deployed unit: **20.2 s/step**, CPU 200%, `RssAnon` 2.6 GB, cgroup
`memory.pressure` 0.00%, cell ≈ 2.1 h (300 steps + 2 eval points), 25 main cells ≈ 40 h
single worker.

Supervision: `/root/lstm_queue_watchdog.sh` (cron `*/15`) relaunches the queue whenever no
unit is active and the full 7×5 grid is incomplete, resets `failed` unit state, and uses
`StartLimitIntervalSec=0` (systemd's default 3-restarts/hour limit would otherwise stop the
arm permanently). `/root/idle_sentinel.py` (cron `*/10`) raises
`/root/lstm_watch_alert.md` when the unit is dead or swap > 3 GB.

---

## 2. The 8× slowdown: torch threads (root cause)

The runner never called `torch.set_num_threads()`, so torch defaulted to one intra-op thread
per core (4) while `OMP_NUM_THREADS=2` — oversubscription, MKL/OpenMP spinning, CPU busy with
no progress.

| torch threads | s / micro-batch (8×256) |
|---|---|
| 1 | 9.73 |
| **2** | **4.95** |
| 4 | **56.06** |

End-to-end: **165 s/step → 20.2 s/step** after `torch.set_num_threads(LSTM_THREADS)` +
`set_num_interop_threads(1)`. Symptom to recognise: process in `R`, CPU ~200%, no memory
pressure, yet throughput ~8× below the benchmark.

---

## 3. Memory: measured envelope

| Quantity | Measured |
|---|---|
| anonymous per worker (training) | 2.6–3.2 GB |
| file-backed (mmap'd packed stream) | ~0.16–0.19 GB touched of 545 MB |
| peak during prepack (single process, serial) | ~4.2 GB RSS |
| swap used per worker under pressure | up to 1.6 GB |

Three global OOM kills occurred before the envelope was understood
(09-19 21:54 worker, 09-19 22:05 `llm-lstm-extra`, 09-20 10:05 worker; each killed process
had anon-rss 3.6 GB).

**Trap — cgroup `MemoryHigh` below real usage**: with `MemoryHigh=3G` and a 3.2 GB worker the
kernel reclaimed continuously; the process sat in `D` state (`wchan=rq_qos_wait`), cgroup
`memory.pressure full` ≈ 50%, CPU ≈ 0%. This looks like a hang but is self-inflicted
throttling. Diagnosis order that worked: `/proc/<pid>/status` (`State`, `RssAnon`, `VmSwap`)
→ `wchan` → cgroup `memory.pressure` → *then* the code.

**Benchmark caution**: a 3-step micro-benchmark reported 2.54 GB peak for a worker that
actually needs 3.2 GB in a long run. Size memory from long runs, not from smoke tests.

---

## 4. Queue-level fixes made during the investigation

1. **prepack skip must include `PACK_VERSION`** — the queue compared against the unversioned
   cache name, so every pass re-packed all cells (~20 min wasted per pass).
2. **Completion marker = full 7×5 grid**, not this pass's `CONDITIONS` — otherwise the
   follow-up pass (negtok/fixed_start) could mark the arm complete while the main 5
   conditions were unfinished.
3. **Serial prepack** — two processes packing concurrently (5.2 GB peak each) triggered the
   22:05 OOM; prepack now runs one cell at a time before workers start.
4. **Never edit a running bash script** — bash reads incrementally; patching
   `lstm_v3_queue.sh` mid-run risks executing misaligned text (observed hazard, restart the
   unit instead).

---

## 5. What this means for the paper

The cpu2 arm remains **budget-limited** (300 × 8192 = 2.46e6 tokens vs the GPT-2 arm's
3.93e8, ≈1/160). The equal-budget architecture-axis claim is carried by the separate GPU LSTM
arm (`LSTM_DEVICE=cuda`, seq 1024 × eff batch 128 × 3000 steps = 3.93e8 tokens, branch
`v2-results-lstm-gpu`). Nothing in this document licenses an equal-budget comparison from the
cpu2 numbers.
