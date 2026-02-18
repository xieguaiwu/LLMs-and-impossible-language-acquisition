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
