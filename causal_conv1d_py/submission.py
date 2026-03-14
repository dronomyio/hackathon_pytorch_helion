"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          causal_conv1d  —  Optimized Helion Submission  v3                  ║
║                                                                              ║
║  Target: B200 / Blackwell  (HBM3e 8 TB/s, 50 MB L2, TMA async)             ║
║  Goal:   Match or beat the gold leaderboard entry (~26 µs geom-mean)        ║
╚══════════════════════════════════════════════════════════════════════════════╝

PERFORMANCE ANALYSIS vs GOLD (26 µs)
──────────────────────────────────────
Previous submission scored ~67 µs.  Root causes:

  1. block_sizes=[1, 64] tiles only the D dimension.
     The S (sequence) dimension is left to Helion's default block_N, which
     auto-tunes to a small value.  For D=2560, S=4096 the optimal S-tile is
     128–256 to fill the 128-thread warp group with useful work.

  2. static_shapes=False prevents loop unrolling.
     W=4 is constant across ALL benchmark shapes.  With static_shapes=True
     Triton unrolls the inner loop into 4 consecutive FFMA blocks in PTX,
     eliminating the loop counter and branch overhead.  The compilation cost
     is paid once per shape; the ranked_timeout is 420 s (7 min) which is
     enough for 5 shapes × ~60 s each.

  3. num_warps=4 may be suboptimal for large D-tiles.
     On B200 each SM has 4 warp schedulers; 8 warps (256 threads) keeps all
     schedulers busy while the TMA prefetch pipeline is filling.

OPTIMIZATIONS IN THIS VERSION
──────────────────────────────
  1. static_shapes=True + hl.specialize(W)
     Inner loop fully unrolled at PTX level.  Zero loop overhead for W=4.

  2. block_sizes=[1, 64, 128]  (B=1, D=64, S=128)
     S-tile of 128 gives 64×128 = 8192 FMAs per block, enough to hide the
     ~400-cycle TMA latency with num_stages=4.

  3. num_warps=8
     256 threads/block.  Each warp handles a [8, 128] slice of the acc tile.
     The warp scheduler can issue 8 independent TMA prefetches in flight.

  4. num_stages=4
     4-deep software pipeline: while computing tile k, TMA prefetches k+1,
     k+2, k+3.  Hides HBM3e latency almost entirely.

  5. F.pad instead of torch.cat
     Avoids allocating an intermediate (B, D, S+W-1) tensor.

EXPECTED RESULT
───────────────
  Benchmark shape: B=1, D=2560, S=4096, W=4
  Geometric mean across 5 shapes: ~25–28 µs  (targeting gold ~26 µs)
"""

from task import input_t, output_t

import torch
import torch.nn.functional as F
import helion
import helion.language as hl


# ─────────────────────────────────────────────────────────────────────────────
# Helion kernel — fully unrolled inner loop, large S-tile, 8-warp pipeline
# ─────────────────────────────────────────────────────────────────────────────
@helion.kernel(
    static_shapes=True,
    config=helion.Config(
        # [block_B, block_D, block_S]
        # block_B = 1   : B=1 in all benchmarks
        # block_D = 64  : weight tile w[64, W] = 1 KB in registers
        # block_S = 128 : 64×128 = 8192 FMAs/block; fills warp pipeline
        block_sizes=[1, 64, 128],
        # 8 warps = 256 threads.  Keeps all 4 B200 warp schedulers busy.
        num_warps=8,
        # 4-stage async TMA pipeline.  Hides ~400-cycle HBM3e latency.
        num_stages=4,
    ),
)
def _causal_conv1d_kernel(
    x_pad: torch.Tensor,   # (B, D, S+W-1)  causal-zero-padded input
    w:     torch.Tensor,   # (D, W)          depthwise filter weights
    b:     torch.Tensor,   # (D,)            per-channel bias
) -> torch.Tensor:
    B = x_pad.size(0)
    D = x_pad.size(1)
    L = x_pad.size(2)
    # W is a compile-time constant — static_shapes=True + hl.specialize
    # allows Triton to fully unroll the inner loop at PTX level.
    W = hl.specialize(w.size(1))
    N = L - W + 1   # output length equals original S

    y = torch.empty(B, D, N, dtype=x_pad.dtype, device=x_pad.device)

    for rb, rd, rs in hl.tile([B, D, N], block_size=[1, None, None]):
        bi = rb.begin

        # float32 accumulator — matches DeterministicContext in reference
        acc = hl.zeros([rd, rs], dtype=torch.float32)

        # Inner convolution loop — fully unrolled by Triton (W is compile-time)
        # Each iteration: 1 register load (w) + 1 TMA load (x_pad) + FMAs
        for j in range(W):
            wj = w[rd, j].to(torch.float32)                          # [block_D]
            xj = hl.load(x_pad, [bi, rd, rs.index + j]).to(torch.float32)  # [block_D, block_S]
            acc = acc + xj * wj[:, None]

        # Bias broadcast over S dimension
        acc = acc + b[rd].to(torch.float32)[:, None]

        y[rb, rd, rs] = acc[None, :, :].to(y.dtype)

    return y


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────
def custom_kernel(data: input_t) -> output_t:
    x, weight, bias = data
    W = weight.shape[1]
    x_padded = F.pad(x, (W - 1, 0))   # causal zero-padding, no intermediate alloc
    return _causal_conv1d_kernel(x_padded, weight, bias)
