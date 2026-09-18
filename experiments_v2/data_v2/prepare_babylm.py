"""BabyLM preparation (v2): sentence-split, split, transform, held-out.

Input: the BabyLM 100M corpus as one or more .txt files (babylm.github.io).
Output: experiments_v2/data_v2/babylm_conditions/{train,test}/<condition>.txt

Protocol notes (preregistration.md section 4):
- sentence splitting uses a conservative regex (or spacy when available) so
  the transformation domain (whitespace words) is well-defined;
- the SAME deterministic sentence-level split (seed 42, 2% held-out) is used
  for every condition; the test set is then perturbed per-condition, so
  cross-condition test PPL comparisons are matched-pair;
- the parity transformation is applied at the sentence level exactly as in
  the SVO pipeline, keeping word-count parity well-defined.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments_v2" / "data_v2"))

from conditions import apply_condition  # noqa: E402

_SENT_RE = re.compile(r"[^.!?]+[.!?]+(?:\"|')?|\S+$")


def split_sentences(text: str) -> list[str]:
    sents = [m.group(0).strip() for m in _SENT_RE.finditer(text)]
    return [s for s in sents if 2 <= len(s.split()) <= 200]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, help="directory with BabyLM .txt files")
    parser.add_argument("--out-dir", default=str(REPO / "experiments_v2" / "data_v2" / "babylm_conditions"))
    parser.add_argument("--test-frac", type=float, default=0.02)
    parser.add_argument("--max-sentences", type=int, default=0, help="0 = all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    texts = []
    for p in sorted(Path(args.raw).glob("*.txt")):
        texts.append(p.read_text(encoding="utf-8", errors="ignore"))
        print(f"read {p.name} ({len(texts[-1])/1e6:.1f}M chars)")
    text = "\n".join(texts)

    try:
        import spacy  # preferred splitter when available

        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "tagger", "lemmatizer"])
        nlp.enable_pipe("sentencizer") if not nlp.has_pipe("sentencizer") else None
        nlp.max_length = 10_000_000
        sentences = []
        for doc in nlp.pipe([text[i : i + 900_000] for i in range(0, len(text), 900_000)]):
            sentences.extend(s.text.strip() for s in doc.sents)
        print("sentence-split with spaCy")
    except Exception:
        sentences = split_sentences(text)
        print("sentence-split with regex (spaCy unavailable)")

    sentences = [s for s in sentences if 2 <= len(s.split()) <= 200]
    if args.max_sentences:
        sentences = sentences[: args.max_sentences]
    print(f"total sentences: {len(sentences)}")

    rng = random.Random(args.seed)
    idx = list(range(len(sentences)))
    rng.shuffle(idx)
    n_test = max(1, int(round(len(sentences) * args.test_frac)))
    test_idx = set(idx[:n_test])
    train = [sentences[i] for i in sorted(set(range(len(sentences))) - test_idx)]
    test = [sentences[i] for i in sorted(test_idx)]

    out_dir = Path(args.out_dir)
    for cond in ["natural", "reversed", "parity_negation"]:
        (out_dir / "train").mkdir(parents=True, exist_ok=True)
        (out_dir / "test").mkdir(parents=True, exist_ok=True)
        for split, data in (("train", train), ("test", test)):
            path = out_dir / split / f"{cond}.txt"
            with open(path, "w", encoding="utf-8") as f:
                for s in apply_condition(data, cond):
                    f.write(s + "\n")
            print(f"{cond:18s} {split}: {len(data):8d} -> {path}")


if __name__ == "__main__":
    main()
