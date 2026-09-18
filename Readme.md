# **Large Language Models and Impossible Language Acquisition: "False Promise" or an Overturn of our Current Perspective towards AI**

## Introduction
**In front of the Repository**

It's a pitch dark. You are likely to be eated by a grue.

>go north

**North Path**

You are now looking at the code repository for our paper "Large Language Models and Impossible Language Acquisition: 'False Promise' or an Overturn of our Current Perspective towards AI".
[arXiv link](https://arxiv.org/abs/2602.08437)

The repository itself contains the training code as well as the preproceeding code for our experiments:

```
LLMs-and-impossible-language-acquisition/
├── baby_data/
│   └── baby_generate_impossible.py
├── colab_train_babydataset.py
├── colab_train_perplexity.py
├── data/
│   ├── generate_impossible.py
│   └── generate_input.py
├── lstm_baby.py
├── lstm_nlp.py
├── Readme.md
├── requirements.txt
├── show_statistics_loss.py
├── show_statistics_perplexity.py
├── T_test.py
├── T_test_baby.py
├── train.py
├── visualize_t_test_baby_results.py
└── visualize_t_test_results.py
```

## Setup
### Clone repository
```
git clone https://github.com/xieguaiwu/LLMs-and-impossible-language-acquisition.git
cd ./LLMs-and-impossible-language-acquisition
pip install -r requirements.txt
```

### Download BabyLM dataset
[Click here to save the Nigerian prince](https://babylm.github.io/)

### Deal with datasets
#### Dataset 1 - home-brew SVO sentences
```
cd ./data
python ./generate_input.py
```

This script generates 'input.txt' under the directory.

```
cp input.txt natural.txt
```

---

For such a smart person like you, I guess it wouldn't hard to figure out how to deal with the rest of the preparation. Good luck!

PS: [Click here to automatically cite the paper](https://www.youtube.com/shorts/11bnjWCDLa0)
---

---

## experiments_v2 — multi-seed replication, controls & probes (2026-09)

`experiments_v2/` contains the v2 experimental suite that addresses the
methodological limitations of the original single-seed study:

- **n≥5 seeds per condition** with per-seed statistical aggregates,
  Holm-Bonferroni correction, Cohen's d + bootstrap CI, and TOST equivalence
  testing for null claims (no more t-tests over serially-correlated
  training steps);
- **marker controls** for parity negation (`fixed_start_neg`, `fixed_end_neg`,
  `<NEG>` special-token variant), **token-unit parity**, and a Kallini-style
  `word_shuffle` reference condition;
- **capacity-matched architecture pair** (gpt2_tiny ≈44M vs lstm_matched ≈39M)
  to unconfound the GPT-2 vs LSTM comparison;
- **behavioral probes** for the parity rule: minimal pairs, violation
  detection, length extrapolation, and a diagnostic hidden-state probe;
- **matched-pair held-out evaluation** on a shared sentence split.

See [`experiments_v2/README.md`](experiments_v2/README.md) and the frozen
preregistration in
[`experiments_v2/preregistration.md`](experiments_v2/preregistration.md)
(read it before interpreting results). CPU smoke tests:
`python3 experiments_v2/tests/test_all.py`.
