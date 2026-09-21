#!/usr/bin/env python3
"""stack_compat.py — stack-portability shims + provenance for the v3 grid.

WHY
---
The registered grid was trained on **torch 2.2.2 + CUDA 12.1** (RTX 3080).
A second host with RTX 5090 (Blackwell, compute capability sm_120) needs
**torch >= 2.7 / cu128** — the older wheels ship no sm_120 kernels — and in that
stack ``torch.cuda.amp.*`` is deprecated (and scheduled for removal).  This
module gives *one* code path for both stacks:

  * ``amp_autocast(enabled)``  — fp16 autocast.  Uses ``torch.autocast`` (present
    since torch 1.10) with ``dtype=torch.float16`` and the default
    ``cache_enabled=True``: the same semantics as ``torch.cuda.amp.autocast()``.
  * ``grad_scaler(enabled)``   — ``GradScaler`` with the historical defaults
    (init_scale 2**16, growth_factor 2.0, backoff 0.5, growth_interval 2000),
    preferring ``torch.amp.GradScaler("cuda", ...)`` on torch >= 2.4.
  * ``pin_numerics()``         — sets the precision flags **explicitly** to the
    values that were already in effect on the 3080 (matmul TF32 off, cuDNN TF32
    on, cuDNN benchmark off, deterministic off), so a stack whose defaults differ
    cannot silently change the training regime.  On the 3080 these assignments
    are no-ops.
  * ``stack_metadata()``       — host / GPU name / capability / torch / CUDA /
    cuDNN / python / transformers, written into every result JSON so that
    cross-host cells stay traceable.

What this module must NOT do: change the registered protocol.  Numbers have to
stay identical on the old stack; any genuine cross-stack drift is measured by a
**bridge cell** (``bridge_check.py``), never assumed — see FALSIFICATION #F9 for
the precedent (two silent protocol deviations invalidated six cells).

Registered as prereg §10c-11 (metadata + explicit no-op numerics pin).
"""

from __future__ import annotations

import platform
import socket

import torch

_NUMERICS_PINNED = False


# --------------------------------------------------------------------- amp ---

def amp_autocast(enabled: bool = True):
    """fp16 autocast context manager that works on torch 2.2 and torch >= 2.7."""
    if not enabled:
        # ``torch.autocast("cpu", enabled=False)`` needs no CUDA and is a no-op.
        return torch.autocast(device_type="cpu", enabled=False)
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.cuda.amp.autocast(dtype=torch.float16)          # very old torch


def grad_scaler(enabled: bool = True):
    """GradScaler with the historical defaults on both stacks."""
    try:                                     # torch >= 2.4 (and torch 2.2.2 warns)
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


# ----------------------------------------------------------------- numerics ---

def pin_numerics(verbose: bool = True) -> dict:
    """Pin the precision/algp flags to the 3080-run values (no-ops there)."""
    global _NUMERICS_PINNED
    flags: dict[str, object] = {}
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        flags["tf32_matmul"] = False
    except Exception as exc:                                  # pragma: no cover
        flags["tf32_matmul"] = f"unset ({exc})"
    try:
        torch.backends.cudnn.allow_tf32 = True
        flags["tf32_cudnn"] = True
    except Exception as exc:                                  # pragma: no cover
        flags["tf32_cudnn"] = f"unset ({exc})"
    try:
        torch.backends.cudnn.benchmark = False
        flags["cudnn_benchmark"] = False
    except Exception as exc:                                  # pragma: no cover
        flags["cudnn_benchmark"] = f"unset ({exc})"
    try:
        torch.backends.cudnn.deterministic = False
        flags["cudnn_deterministic"] = False
    except Exception as exc:                                  # pragma: no cover
        flags["cudnn_deterministic"] = f"unset ({exc})"
    # torch >= 2.9 exposes the fp32_precision API; keep it in step with
    # allow_tf32=False ("ieee") so a future default flip cannot leak in.
    try:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        flags["fp32_precision_matmul"] = "ieee"
    except Exception:
        pass
    _NUMERICS_PINNED = True
    if verbose:
        print(f"[stack] numerics pinned: {flags}", flush=True)
    return flags


def numerics_flags() -> dict:
    """Read back the effective flags (for the result JSON)."""
    out: dict[str, object] = {}
    try:
        out["tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
    except Exception:
        out["tf32_matmul"] = None
    try:
        out["tf32_cudnn"] = bool(torch.backends.cudnn.allow_tf32)
    except Exception:
        out["tf32_cudnn"] = None
    try:
        out["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    except Exception:
        out["cudnn_benchmark"] = None
    try:
        out["fp32_precision_matmul"] = getattr(
            torch.backends.cuda.matmul, "fp32_precision", None)
    except Exception:
        out["fp32_precision_matmul"] = None
    return out


# ---------------------------------------------------------------- provenance ---

def stack_metadata() -> dict:
    """Host + stack identity written into every result JSON (cross-host trace)."""
    md: dict[str, object] = {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        md["cudnn"] = torch.backends.cudnn.version()
    except Exception:
        md["cudnn"] = None
    try:
        import numpy as _np
        md["numpy"] = _np.__version__
    except Exception:
        md["numpy"] = None
    try:
        import transformers as _tf
        md["transformers"] = _tf.__version__
    except Exception:
        md["transformers"] = None
    if torch.cuda.is_available():
        try:
            idx = torch.cuda.current_device()
            md["device_name"] = torch.cuda.get_device_name(idx)
            cap = torch.cuda.get_device_capability(idx)
            md["capability"] = f"{cap[0]}.{cap[1]}"
        except Exception:
            md["device_name"] = None
            md["capability"] = None
        md["device_count"] = torch.cuda.device_count()
    md.update(numerics_flags())
    md["numerics_pinned"] = _NUMERICS_PINNED
    md["stack_id"] = f"torch{torch.__version__}-cu{torch.version.cuda}"
    return md