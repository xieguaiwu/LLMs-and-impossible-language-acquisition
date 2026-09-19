# Packing byte-identity evidence (LSTM arm vs GPT-2 arm)

Recorded 2026-09-19 on cpu2 (HEAD `e7047cd`, protocol base `12f22f9` = tail-drop semantics).
Owner suggestion #3 (llm-impossible session 01a0b7de): keep the proof next to the runner so a
reviewer can check "why do you claim both arms pack token-for-token identically".

## What is compared

- **Theirs**: `train_exp1.load_packed_dataset(condition, seed)` — the GPT-2 arm's packer
  (shuffle sentences with `np.random.default_rng(seed)`, join with EOS, chunk into 1024,
  **drop the trailing partial block** since 12f22f9).
- **Mine**: `train_exp1_lstm.packed_blocks(condition, seed)` — numpy int32 twin
  (`_sentence_stream` → same shuffle over an index array → ragged gather → re-chunk to 256).
- Comparison is on the **token stream** (concatenated blocks vs concatenated windows),
  not on the container: 3 seeds, 2 real corpus files truncated to 300 sentences each.

## Invariants asserted by the LSTM packer (not smoke-tested)

```
window width      == SEQ_LEN
window count      == kept // SEQ_LEN
token conservation: kept == total - (total % SEQ_LEN)   # tail dropped, nothing invented
no partial window survives;  n_full > 0
```

## Cache-version tag (reproducibility)

`PACK_VERSION=v2` is part of every cache filename
(`results_lstm/cache/<cond>_seed<N>_seq256_v2.npy`), so caches written under the old
semantics (padded tail window) can never be silently reused after a protocol change.

### 复现命令
```bash
KALLINI_DATA_PATH=/root/kallini_data python3 experiments_v2/kallini_repro/packing_equivalence_check.py
```

### 输出（2026-09-19 20:57:11，HEAD e7047cd）
```
seed=  0 sentences=    600 tokens=     6307 kept=     6144 (dropped tail= 163) len_match=True identical=True
seed= 14 sentences=    600 tokens=     6307 kept=     6144 (dropped tail= 163) len_match=True identical=True
seed= 41 sentences=    600 tokens=     6307 kept=     6144 (dropped tail= 163) len_match=True identical=True
PASS: LSTM packer reproduces the GPT-2 arm's token stream exactly (same tail-drop semantics, all windows full-width)
```

## Scope note

The check runs on a truncated corpus copy (fast, seconds). It is an equivalence proof of the
*algorithm*; the full-corpus invariant is asserted inside every real run at prepack time
(`packed_blocks`), so a broken packer fails at cell start instead of mid-training.
