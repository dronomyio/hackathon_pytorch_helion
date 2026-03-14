"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   FP8 Per-Group Quantization  —  Optimized Helion Submission  v2            ║
║                                                                              ║
║  Target: B200 / Blackwell  (HBM3e 8 TB/s, 50 MB L2, TMA async)             ║
╚══════════════════════════════════════════════════════════════════════════════╝

WHAT THE KERNEL DOES
─────────────────────
Input:  x    [T, H]       float32
Output: x_q  [T, H]       float32  (quantized values, clamped to FP8 E4M3 range)
        x_s  [T, H//G]    float32  (per-group scales)

For each token t and group g (group_size = H // num_groups):
  1. absmax = max(|x[t, g*gs : (g+1)*gs]|)
  2. scale  = max(absmax, eps) / 448.0
  3. x_q    = clamp(x / scale, -448.0, 448.0)

WHY THE BASELINE IS SLOW
─────────────────────────
  Bug 1 — WRONG MATH: (amax1+amax2+amax3)/3 and (q1+q2+q3)/3
    Loads x THREE times with the same indices and averages — wrong answer
    AND 3× memory bandwidth waste.

  Bug 2 — WRONG TILING: tiles over (T*G) rows of shape [group_size]
    Each row = one group, so each thread block handles 1 token × 1 group.
    With T=4096, G=56 → 229,376 tiny blocks. Grid-launch overhead dominates.

  Bug 3 — static_shapes=True with 5 different (T,H,group_size) combos
    Compiles 5 separate PTX binaries. Exceeds the 420s ranked_timeout.

WHAT THIS KERNEL DOES DIFFERENTLY
───────────────────────────────────
  Fix 1 — CORRECT single-pass: absmax → scale → quantize in registers
  Fix 2 — TILE [block_T=16, block_G=1]: 16 tokens × 1 group per block
           16 × group_size FMAs per block; good occupancy, low overhead
  Fix 3 — static_shapes=False: ONE compilation for all shapes
  Fix 4 — num_warps=4, num_stages=2: 128 threads + async TMA prefetch
"""

from task import input_t, output_t

import torch
import helion
import helion.language as hl

FP8_MAX = 448.0
FP8_MIN = -448.0
FP8_EPS = 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# Helion kernel — fused absmax + scale + quantize, one pass over x
# ─────────────────────────────────────────────────────────────────────────────
@helion.kernel(
    static_shapes=False,
    config=helion.Config(
        # [block_T, block_G]
        # block_T = 16 : 16 tokens per block
        # block_G = 1  : 1 group per block; group_size elements loaded as a slice
        block_sizes=[16, 1],
        num_warps=4,    # 128 threads — good occupancy for this memory-bound kernel
        num_stages=2,   # 2-stage async prefetch hides HBM latency
    ),
)
def _fp8_quant_kernel(
    x:          torch.Tensor,   # (T, H)   input float32
    x_q:        torch.Tensor,   # (T, H)   output quantized float32
    x_s:        torch.Tensor,   # (T, G)   output scales float32
    group_size: int,
) -> None:
    T = x.size(0)
    G = x_s.size(1)   # num_groups = H // group_size

    for rt, rg in hl.tile([T, G], block_size=[None, None]):
        # ── Load input tile ──────────────────────────────────────────────────
        col_start = rg.index * group_size
        # x_tile: [block_T, group_size]
        x_tile = x[rt, col_start : col_start + group_size].to(torch.float32)

        # ── Per-group absmax (warp reduction, stays in registers) ────────────
        absmax = hl.reduce(x_tile.abs(), dim=1, op="max")   # [block_T]
        absmax = absmax.clamp(min=FP8_EPS)

        # ── Scale ─────────────────────────────────────────────────────────────
        scale = absmax / FP8_MAX   # [block_T]

        # ── Quantize + clamp ──────────────────────────────────────────────────
        q = (x_tile / scale[:, None]).clamp(FP8_MIN, FP8_MAX)

        # ── Store outputs ─────────────────────────────────────────────────────
        x_q[rt, col_start : col_start + group_size] = q.to(x_q.dtype)
        x_s[rt, rg] = scale.to(x_s.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────
def custom_kernel(data: input_t) -> output_t:
    x, x_q, x_s = data
    H = x.shape[1]
    G = x_s.shape[1]
    group_size = H // G

    _fp8_quant_kernel(x, x_q, x_s, group_size)
    return x_q, x_s
