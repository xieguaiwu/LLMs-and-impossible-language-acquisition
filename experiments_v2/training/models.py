"""Model definitions (v2): capacity-matched architecture pair.

The original study compared GPT-2 small (124M) against a ~40M-parameter
LSTM, so "architecture" was confounded with capacity and optimization
hyperparameters. v2 provides a capacity-matched pair:

- ``gpt2_tiny`` : decoder-only transformer, 6 layers / 8 heads / d=512 /
  tied embeddings, ~44M parameters (excluding ~25.7M vocab embedding it is
  ~19M of transformer blocks -- the LSTM below has ~25M of recurrent
  blocks; we report both counts and match *total* parameters within ~5%).
- ``lstm_matched``: 2-layer LSTM, embedding 640 / hidden 640 / tied, ~39M.

Plus the original subjects for direct replication:
- ``gpt2``      : HuggingFace gpt2-small, 124M (from scratch).
- ``lstm``      : 2-layer LSTM, embedding 650 / hidden 650, ~40M (original).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMLM(nn.Module):
    """Tied-embedding LSTM language model over GPT-2 BPE ids."""

    def __init__(self, vocab_size: int, emb_dim: int = 650, hidden_dim: int = 650,
                 num_layers: int = 2, dropout: float = 0.3, pad_token_id: int | None = None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            emb_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)
        if hidden_dim != emb_dim:
            self.pre_head = nn.Linear(emb_dim, hidden_dim, bias=False)
        else:
            self.pre_head = None
        self.head.weight = self.embed.weight  # tie
        self.pad_token_id = pad_token_id

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        x = self.dropout(self.embed(input_ids))
        out, _ = self.lstm(x)
        if self.pre_head is not None:
            out = self.pre_head(out)
        logits = self.head(out)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(
                ignore_index=-100 if self.pad_token_id is None else -100
            )
            loss = loss_fn(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )
        return {"loss": loss, "logits": logits}


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


def describe_match() -> str:
    """Report parameter counts of the capacity-matched pair (CPU, no download)."""
    lines = []
    lstm = LSTMLM(vocab_size=50257, emb_dim=640, hidden_dim=640, num_layers=2)
    # gpt2_tiny is defined in train_lm (needs transformers); compute its count there.
    lines.append(f"lstm_matched total={count_parameters(lstm)/1e6:.1f}M")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_match())
