#!/usr/bin/env python3
"""Shim tagger for the Kallini et al. (2024) data pipeline.

Their ``data/tag.py`` runs Stanza (tokenize+POS+lemma, GPU) over BabyLM files
and writes ``*_parsed.json``. ``data/perturb.py`` reads those JSONs and the
Shuffle/Reverse perturbations only use ``sent["sent_text"]`` (POS/lemma fields
are consumed exclusively by the *Hop perturbations, which are OUT OF SCOPE for
this reproduction round).

This shim produces the exact same JSON structure with pure-Python sentence
segmentation (regex), no stanza dependency. Sentence-boundary differences vs
stanza default_accurate are a documented, minor deviation for the
Shuffle/Reverse languages.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_SENT_RE = re.compile(r"[^.!?]*[.!?]+(?:\"|')?\s*|\S[^.!?]*$")


def split_sentences(text: str) -> list[str]:
    sents = [m.group(0).strip() for m in _SENT_RE.finditer(text)]
    return [s for s in sents if s.strip()]


def tag_file(path: Path) -> None:
    lines = [l.strip() for l in path.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
    BATCH = 5000
    line_annotations = []
    n_sents = 0
    for i in range(0, len(lines), BATCH):
        batch_text = " ".join(lines[i : i + BATCH])
        sents = split_sentences(batch_text)
        anns = [{"sent_text": s, "words": []} for s in sents]
        n_sents += len(anns)
        line_annotations.append({"sent_annotations": anns})
    out = path.parent / (path.stem + "_parsed.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(line_annotations, f)
    print(f"tagged {path.name}: {n_sents} sentences -> {out.name}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+", help="raw BabyLM text files")
    args = p.parse_args()
    for f in args.files:
        tag_file(Path(f))
