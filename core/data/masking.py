from __future__ import annotations
import torch
from torch import Tensor

def sample_block_masks(
    seq_len: int,
    n_target_blocks: int,
    target_block_frac: tuple[float, float],
    context_keep_frac: float,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, list[Tensor]]:
    g = generator
    lo, hi = target_block_frac
    target_positions: set[int] = set()
    target_blocks: list[Tensor] = []
    for _ in range(n_target_blocks):
        frac = lo + (hi - lo) * torch.rand(1, generator=g).item()
        blen = max(1, int(round(frac * seq_len)))
        start_max = seq_len - blen
        start = int(torch.randint(0, start_max + 1, (1,), generator=g).item())
        block = torch.arange(start, start + blen, dtype=torch.long)
        target_blocks.append(block)
        target_positions.update(block.tolist())
    all_positions = set(range(seq_len))
    context_candidates = sorted(all_positions - target_positions)
    context_candidates_t = torch.tensor(context_candidates, dtype=torch.long)
    keep = max(1, int(round(context_keep_frac * len(context_candidates_t))))
    if keep < len(context_candidates_t):
        perm = torch.randperm(len(context_candidates_t), generator=g)[:keep]
        context_idx = context_candidates_t[perm.sort().values]
    else:
        context_idx = context_candidates_t
    return context_idx, target_blocks