"""Per-run metric recording and summary computation (v2).

Every training run writes one JSON file with:
- the raw per-step series (backward compatible with the old files), and
- a ``summary`` block of per-run scalars used as the unit of statistical
  analysis (one scalar per seed -- the fix for the time-series pseudo-
  replication problem documented in statistics/EXPDESIGN.md section 4.1).

Summary metrics
---------------
final_loss       mean loss over the last 10% of logged steps
min_loss         minimum logged loss
final_ppl        exp(final_loss)
min_ppl          exp(min_loss)
auc_loss         mean loss over all logged steps
convergence_step first step whose trailing rolling mean is within 2% of
                 the final rolling mean (earlier = faster convergence)
test_loss        cross-entropy on the condition-perturbed held-out test set
test_ppl         exp(test_loss)
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

CONVERGENCE_TOL = 0.02
FINAL_WINDOW_FRAC = 0.10


def summarize_series(losses: list[float], total_steps: int | None = None) -> dict:
    """Compute per-run scalar summary from a per-step loss series."""
    if not losses:
        return {}
    n = len(losses)
    w = max(1, int(round(n * FINAL_WINDOW_FRAC)))
    final_loss = sum(losses[-w:]) / w
    min_loss = min(losses)
    auc_loss = sum(losses) / n

    rolling: list[float] = []
    win = max(1, n // 10)
    for i in range(n):
        lo = max(0, i - win + 1)
        rolling.append(sum(losses[lo : i + 1]) / (i - lo + 1))
    target = rolling[-1]
    threshold = target * (1 - CONVERGENCE_TOL) if target > 0 else target * (1 + CONVERGENCE_TOL)
    convergence_step = next(
        (i + 1 for i, v in enumerate(rolling) if v <= threshold), n
    )

    step_of = lambda i: int(round((i + 1) * (total_steps or n) / n))
    return {
        "final_loss": round(final_loss, 6),
        "min_loss": round(min_loss, 6),
        "final_ppl": round(math.exp(final_loss), 6),
        "min_ppl": round(math.exp(min_loss), 6),
        "auc_loss": round(auc_loss, 6),
        "convergence_step": step_of(convergence_step - 1),
        "convergence_frac": round(convergence_step / n, 4),
    }


def write_run_json(
    out_path: Path,
    *,
    run_id: str,
    experiment: str,
    model: str,
    dataset: str,
    condition: str,
    seed: int,
    hyperparameters: dict,
    losses: list[float],
    test_loss: float | None,
    total_steps: int,
    training_time_seconds: float | None = None,
    extra: dict | None = None,
) -> dict:
    """Write one run's JSON (old-format-compatible) and return the record."""
    summary = summarize_series(losses, total_steps)
    summary["test_loss"] = None if test_loss is None else round(float(test_loss), 6)
    if summary.get("test_loss") is not None:
        summary["test_ppl"] = round(math.exp(summary["test_loss"]), 6)

    record = {
        "run_id": run_id,
        "experiment": experiment,
        "model": model,
        "dataset": dataset,
        "condition": condition,
        "seed": seed,
        "hyperparameters": hyperparameters,
        "losses": losses,
        "final_loss": losses[-1] if losses else None,  # old-format field
        "summary": summary,
        "total_steps": total_steps,
        "training_time_seconds": training_time_seconds,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        record.update(extra)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return record


def load_runs(runs_dir: Path) -> list[dict]:
    records = []
    for p in sorted(runs_dir.glob("*.json")):
        with open(p, encoding="utf-8") as f:
            records.append(json.load(f))
    return records
