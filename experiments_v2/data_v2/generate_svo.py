"""Clean SVO corpus generator (v2).

Fixes a hidden bug in the original ``data/generate_input.py``: that script
wrote ``Original: <sentence>`` lines into ``input.txt`` together with the
plain sentences, so the downstream ``generate_impossible.py`` treated the
"Original:" prefix as part of the sentence and reversed corpora contained
artifacts such as ``.book the like can man The :Original``.

v2 writes one plain sentence per line, with a fixed RNG seed so every run
regenerates the identical corpus. Grammar template follows the original
(auxiliaries, progressive, passive, perfect) to stay comparable.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

SUBJECTS_SINGULAR = [
    "The man", "The woman", "The boy", "The girl", "The student",
    "The teacher", "The doctor", "The chef", "The artist", "The driver",
    "The cat", "The dog", "The bird", "The horse", "The rabbit",
    "The child", "The parent", "The friend", "The neighbor", "The worker",
    "The player", "The singer", "The dancer", "The writer", "The reader",
]

SUBJECTS_PLURAL = [
    "The men", "The women", "The boys", "The girls", "The students",
    "The teachers", "The doctors", "The chefs", "The artists", "The drivers",
    "The cats", "The dogs", "The birds", "The horses", "The rabbits",
    "The children", "The parents", "The friends", "The neighbors", "The workers",
    "The players", "The singers", "The dancers", "The writers", "The readers",
]

PRONOUNS = ["I", "You", "He", "She", "We", "They", "It"]

AUXILIARIES = ["can", "will", "must", "should", "may", "might", "would", "could", "shall"]

BASE_VERBS = [
    "like", "love", "see", "hear", "know", "find", "take", "give",
    "make", "buy", "sell", "read", "write", "draw", "paint",
    "cook", "eat", "drink", "play", "watch", "study", "learn",
    "teach", "help", "call", "ask", "answer", "open", "close",
    "build", "create", "design", "develop", "understand", "explain",
    "remember", "forget", "enjoy", "hate", "prefer", "want", "need",
]

PAST_PARTICIPLES = [
    "liked", "loved", "seen", "heard", "known", "found", "taken", "given",
    "made", "bought", "sold", "read", "written", "drawn", "painted",
    "cooked", "eaten", "drunk", "played", "watched", "studied", "learned",
    "taught", "helped", "called", "asked", "answered", "opened", "closed",
    "built", "created", "designed", "developed", "understood", "explained",
    "remembered", "forgotten", "enjoyed", "hated", "preferred", "wanted", "needed",
]

PRESENT_PARTICIPLES = [
    "liking", "loving", "seeing", "hearing", "knowing", "finding", "taking", "giving",
    "making", "buying", "selling", "reading", "writing", "drawing", "painting",
    "cooking", "eating", "drinking", "playing", "watching", "studying", "learning",
    "teaching", "helping", "calling", "asking", "answering", "opening", "closing",
    "building", "creating", "designing", "developing", "understanding", "explaining",
    "remembering", "forgetting", "enjoying", "hating", "preferring", "wanting", "needing",
]

OBJECTS = [
    "the book", "the pen", "the apple", "the car", "the house",
    "the ball", "the computer", "the phone", "the music", "the movie",
    "the food", "the water", "the coffee", "the tea", "the cake",
    "the picture", "the song", "the game", "the toy", "the flower",
    "the tree", "the sun", "the moon", "the sky", "the sea",
    "the dog", "the cat", "the bird", "the fish", "the horse",
    "the answer", "the question", "the problem", "the solution", "the idea",
    "the work", "the job", "the task", "the project", "the plan",
    "the painting", "the story", "the letter", "the message", "the news",
    "the lesson", "the class", "the school", "the garden", "the park",
]


def _be_verb(subject: str) -> str:
    if subject == "I":
        return "am"
    if subject in ("He", "She", "It"):
        return "is"
    if subject in ("You", "We", "They"):
        return "are"
    if subject.startswith("The ") and subject not in SUBJECTS_PLURAL:
        return "is"
    return "are"


def generate_svo_sentences(count: int, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    sentences: list[str] = []
    for _ in range(count):
        stype = rng.choice(["pronoun", "singular", "plural"])
        aux_type = rng.choice(["modal", "be_progressive", "be_passive", "have_perfect"])

        if stype == "pronoun":
            subject = rng.choice(PRONOUNS)
        elif stype == "singular":
            subject = rng.choice(SUBJECTS_SINGULAR)
        else:
            subject = rng.choice(SUBJECTS_PLURAL)

        obj = rng.choice(OBJECTS)
        verb = rng.choice(BASE_VERBS)
        verb_past = rng.choice(PAST_PARTICIPLES)
        verb_present = rng.choice(PRESENT_PARTICIPLES)
        be = _be_verb(subject)
        have = "has" if be == "is" else "have"

        if aux_type == "modal":
            sentence = f"{subject} {rng.choice(AUXILIARIES)} {verb} {obj}."
        elif aux_type == "be_progressive":
            be2 = be if rng.random() < 0.5 else ("was" if be in ("is", "am") else "were")
            sentence = f"{subject} {be2} {verb_present} {obj}."
        elif aux_type == "be_passive":
            be2 = be if rng.random() < 0.5 else ("was" if be in ("is", "am") else "were")
            sentence = f"{subject} {be2} {verb_past} {obj}."
        else:
            have2 = have if rng.random() < 0.5 else "had"
            sentence = f"{subject} {have2} {verb_past} {obj}."

        sentences.append(sentence)
    return sentences


def write_lines(sentences: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="experiments_v2/data_v2/svo_sentences.txt")
    args = parser.parse_args()

    sentences = generate_svo_sentences(args.count, args.seed)
    out = Path(args.out)
    write_lines(sentences, out)
    print(f"wrote {len(sentences)} sentences -> {out}")


if __name__ == "__main__":
    main()
