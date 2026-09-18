# Kallini et al. (2024) 设计要素笔记 + 本项目 v2 实证信号（供 v3 设计 agent 使用）

来源：arXiv:2401.06416v1 全文 + 官方 repo jkallini/mission-impossible-language-models 逐行审读。
本文件是"复现方笔记"，供论文 v3 实验设计引用。

## 1. 语言类结构与控制组（他们最核心的设计原则）

三个语言类，**每类自带一个 control 语言**（这是我们要吸收的关键设计）：

- \*Shuffle 类（control = NoShuffle = 原句 token 化）：
  - NondeterministicShuffle（每句独立随机洗牌，不可逆）
  - DeterministicShuffle(s1/s2/s3)（按 token 长度分桶确定性洗牌，可逆但非语言性）
  - LocalShuffle(w=3/5/10)（局部窗口确定性洗牌）
  - EvenOddShuffle（偶位在前奇位在后）
- \*Reverse 类（control = NoReverse：原句 + 随机位置插入 marker token R）：
  - PartialReverse（marker 后反转）
  - FullReverse（全反转）
  - 关键：R marker 在所有 Reverse 语言中**位置相同**，用于熵控制
- \*Hop 类（control = NoHop：动词词元化 + 紧跟 S/P 性数标记）：
  - TokenHop：S/P 放动词后 4 个 **token** 处
  - WordHop：放 4 个 **词** 处（跳过标点）
  - 计数规则类

## 2. 训练与评测协议

- GPT-2 small 从零训练，BabyLM 100M 词（9.69M 句 shuffle/reverse 类；8.43M 句 hop 类，经过滤）
- 3000 优化步 ≈ 11 epochs；**有效 batch 512**（seq 1024）；lr 线性 warmup 300 步 → 6e-4；
  稳定性 flag：reorder_and_upcast_attn + scale_attn_by_inverse_layer_idx
- 词表：shuffle 50257；reverse +marker R = 50258；hop +S/P = 50259
- 评测：**测试集 = BabyLM test 抽 10,000 句按各语言扰动**，逐句 ppl，报告**几何平均**，
  checkpoint 每 100 步全 ladder，5 seeds（[0,14,41,53,96]）误差棒
- 数据打包：句 token 序列按 seed 用 numpy rng 洗序 → EOS 拼接 → 1024 分块
- Hop 类过滤：marker 放不进 4 hop 的句子从全部条件剔除（保证跨条件句子集相同）

## 3. 三个实验

1. **Exp 1（ppl 连续统）**：语言越不可能 ppl 越高；排序（图2）：
   NoShuffle < LocalShuffle(w小) ≈ EvenOdd < Local(w大) < DeterministicShuffle < NondeterministicShuffle；
   Reverse 类接近 control（Partial 略好于 Full）；**Hop 类 ppl 差异极小**（故引入 Exp2/3）
2. **Exp 2（surprisal）：** 构造最小对：S/P marker 在合规 vs 非法位置；比较 S(marker) 与
   S(下一token) 的差值随训练的轨迹 → NoHop（最自然）的 marker-surprisal-delta 最高，
   TokenHop > WordHop（token 计数比词计数更可学）
3. **Exp 3（causal abstraction）**：interleaving heads 等内部机制分析

## 4. 他们的重要设计原则（v3 必须继承）

a. 语言按"不可能性连续统"组织，每类带 control；
b. 评测用 **held-out 扰动测试集的几何平均 ppl × checkpoint 阶梯**（不是 train loss！训练曲线只作过程参考）；
c. marker 熵控制：凡插入特殊 token 的语言，其 control 也插入同分布 marker；
d. 过滤函数保证跨条件句子集一致（filter 不通过的句子从所有条件剔除）；
e. 5 seeds + 95% CI；评测在多个 checkpoint 观察学习过程而非只看终点。

## 5. 本项目 v2 迄今实证信号（2026-09-18 夜，clean SVO 玩具语料 10k 句）

- 复制协议（1410步/batch4）：natural test 1.918 ≈ reversed 1.915（无差距，**与原论文 Exp1 相反**）
- word_shuffle test 3.11 —— 真随机洗牌的不可能性信号强 ✓（管线有效）
- 否定 marker 类反而更容易：fixed_start 1.677 / fixed_end 1.677 / parity 1.668（< natural 1.918）
- negtok（<NEG> 单 token 版）1.606 < parity 1.668（marker 干净化更易）
- H7 扩展臂（3× 步数）：train 1.71→0.55 但 test 1.92→2.79 —— 玩具语料过拟合，test 失效
  → **H7 必须迁到 BabyLM**
- parity_tok ≡ parity：模板词全单 BPE → tok-奇偶≡词奇偶，条件在 SVO 空洞，须在 BabyLM 上才有意义
- H8 污染臂（复现原论文语料 bug）：natural_polluted train 1.71→1.36（方向支持"原 Exp1 数字是
  语料重复伪影"），全量 15 runs 待出

## 6. 本论文（Ziyan Wang v2→v3）的独特贡献点（相对 Kallini）

- 架构对比：GPT-2 small vs **容量配对 LSTM**（Kallini 只用 GPT-2）——本文核心增量
- 哲学层：Chomsky "LLM 无法区分可能/不可能"前提的经验检验 + 功能主义/经验主义范式论证
- marker 对照族（fixed_start/fixed_end/negtok）作为 parity 类的控制组（对应 Kallini 的
  NoReverse marker 控制思想）
- 行为探针：parity 最小对 / 长度外推 / 隐状态诊断探针（对应 Kallini Exp2 的 surprisal 法）
- 预算依赖（H7：优势是否随算力涌现）
