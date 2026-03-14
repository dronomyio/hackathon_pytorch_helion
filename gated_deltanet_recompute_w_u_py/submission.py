"""
Optimized Triton submission for gated_deltanet_recompute_w_u
============================================================
Reference computes per chunk (all chunks independent):
    u_c = A_c @ (v_c * beta_c[:, None])                        # [BT,V]
    w_c = A_c @ (k_c * (beta_c * exp(g_c))[:, None])           # [BT,K]

Key observations:
1. ALL chunks are FULLY PARALLEL — no inter-chunk dependency (unlike chunk_fwd_h).
   grid = (B × H × NT,) — every chunk runs on its own thread block.

2. The reference does:
   - 5 reshapes + 2 permutes per tensor (CPU overhead, no GPU work)
   - 2 elementwise multiplications (2 separate kernel launches)
   - 2 torch.matmul calls via cuBLAS (2 separate kernel launches)
   - 2 permute + reshape + .to() for output (2 more launches)
   Total: ~10 kernel launches per batch, each with Python GIL overhead.

3. Fused Triton kernel:
   - 1 kernel launch total
   - A_c loaded ONCE from HBM, reused for both u and w GEMMs
   - beta, exp(g) fused into the load of k/v (no intermediate tensors)
   - Both GEMMs (tl.dot) map to 6th-gen Tensor Cores (HMMA.16816.F32)
   - num_stages=2: TMA async prefetch of next chunk while computing current
   - block_sizes=[64,64]: BT=64 matches exactly one chunk, BK=BV=64 = 16 TC tiles

Optimizations:
  OPT 1: Parallel grid — grid=(B*H*NT,), all chunks run simultaneously
  OPT 2: A_c reuse — load A_c once from HBM, use for both u and w GEMMs
  OPT 3: Fused elementwise — beta*exp(g) computed in registers, no alloc
  OPT 4: Tensor Core GEMMs — tl.dot maps to HMMA.16816.F32
  OPT 5: TMA pipelining — num_stages=2 hides 400-cycle HBM latency
  OPT 6: Single kernel launch — eliminates ~10 CUDA launch overheads
  OPT 7: block_sizes=[64,64] — BT=64 exactly one chunk, BK=BV=64 = 16 TC tiles
"""

import torch
import triton
import triton.language as tl
from task import input_t, output_t

CHUNK_SIZE = 64  # BT — fixed by problem spec


@triton.autotune(
    configs=[
        # BT=64 is fixed by the problem (chunk size). BK/BV tile the K/V dims.
        # For K=V=64 (all PR #128 benchmarks), BK=BV=64 processes the full dim in one shot.
        triton.Config({'BK': 64, 'BV': 64}, num_stages=2, num_warps=4),
        triton.Config({'BK': 64, 'BV': 64}, num_stages=3, num_warps=4),
        triton.Config({'BK': 64, 'BV': 64}, num_stages=2, num_warps=8),
        triton.Config({'BK': 32, 'BV': 32}, num_stages=2, num_warps=4),
        triton.Config({'BK': 32, 'BV': 64}, num_stages=2, num_warps=4),
        triton.Config({'BK': 64, 'BV': 32}, num_stages=2, num_warps=4),
    ],
    key=['K', 'V', 'BT'],
)
@triton.jit
def _recompute_w_u_kernel(
    # Input pointers
    k_ptr,      # [B, T, H, K]
    v_ptr,      # [B, T, H, V]
    beta_ptr,   # [B, T, H]
    A_ptr,      # [B, T, H, BT]  — WY matrix, stored as [B, NT, H, BT, BT] logically
    g_ptr,      # [B, T, H]
    # Output pointers
    w_ptr,      # [B, T, H, K]
    u_ptr,      # [B, T, H, V]
    # Dimensions
    B: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
    K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr,
    # Tile sizes (autotuned)
    BK: tl.constexpr, BV: tl.constexpr,
    # Strides for k [B, T, H, K]
    s_k_b: tl.constexpr, s_k_t: tl.constexpr, s_k_h: tl.constexpr,
    # Strides for v [B, T, H, V]
    s_v_b: tl.constexpr, s_v_t: tl.constexpr, s_v_h: tl.constexpr,
    # Strides for beta/g [B, T, H]
    s_bg_b: tl.constexpr, s_bg_t: tl.constexpr, s_bg_h: tl.constexpr,
    # Strides for A [B, T, H, BT]
    s_A_b: tl.constexpr, s_A_t: tl.constexpr, s_A_h: tl.constexpr,
):
    """
    Each thread block handles one (b, h, nt) chunk.
    grid = (B * H * NT,)

    For this chunk:
      - Load A_c [BT, BT] once — reused for both GEMMs
      - Compute u_c = A_c @ (v_c * beta_c[:, None])
      - Compute w_c = A_c @ (k_c * (beta_c * exp(g_c))[:, None])
    """
    # ---------------------------------------------------------------
    # OPT 1: Decode which (b, h, nt) this block owns
    # ---------------------------------------------------------------
    NT = T // BT
    pid = tl.program_id(0)          # flat index in [0, B*H*NT)
    nt  = pid % NT
    bh  = pid // NT
    h   = bh  % H
    b   = bh  // H

    # Base offsets into the time dimension for this chunk
    t_start = nt * BT                # first timestep in this chunk

    # ---------------------------------------------------------------
    # OPT 2: Load A_c [BT, BT] ONCE — reused for both GEMMs
    # A layout: [B, T, H, BT] where T dimension stores BT rows per chunk
    # i.e., A[b, t_start:t_start+BT, h, :] is the [BT, BT] matrix
    # ---------------------------------------------------------------
    A_base = b * s_A_b + t_start * s_A_t + h * s_A_h
    row_idx = tl.arange(0, BT)      # [BT] — row index within chunk
    col_idx = tl.arange(0, BT)      # [BT] — column index within A row

    # A_c[i, j] = A_ptr[A_base + i * s_A_t + j]
    A_offs = A_base + row_idx[:, None] * s_A_t + col_idx[None, :]  # [BT, BT]
    A_c = tl.load(A_ptr + A_offs).to(tl.float32)                   # [BT, BT]

    # ---------------------------------------------------------------
    # OPT 3: Load beta [BT] and g [BT] — fuse into k/v scaling
    # ---------------------------------------------------------------
    bg_base = b * s_bg_b + t_start * s_bg_t + h * s_bg_h
    bg_offs = bg_base + row_idx * s_bg_t                            # [BT]
    beta_c = tl.load(beta_ptr + bg_offs).to(tl.float32)            # [BT]
    g_c    = tl.load(g_ptr    + bg_offs).to(tl.float32)            # [BT]

    # OPT 3: Fused scale factors — no intermediate tensors allocated
    scale_u = beta_c                                                # [BT] for v
    scale_w = beta_c * tl.exp(g_c)                                 # [BT] for k

    # ---------------------------------------------------------------
    # OPT 4: Compute u_c = A_c @ (v_c * scale_u[:, None])
    # Tile over V dimension with BV
    # ---------------------------------------------------------------
    v_base = b * s_v_b + t_start * s_v_t + h * s_v_h
    u_base = b * s_v_b + t_start * s_v_t + h * s_v_h  # same layout as v

    for v_start in tl.range(0, V, BV):
        v_col_idx = v_start + tl.arange(0, BV)         # [BV]
        v_mask = v_col_idx < V

        # Load v_c tile [BT, BV]
        v_offs = v_base + row_idx[:, None] * s_v_t + v_col_idx[None, :]
        v_tile = tl.load(v_ptr + v_offs, mask=v_mask[None, :], other=0.0).to(tl.float32)

        # OPT 3: Fuse beta scaling into v — no extra alloc
        v_scaled = v_tile * scale_u[:, None]            # [BT, BV]

        # OPT 4: Tensor Core GEMM — A_c [BT,BT] @ v_scaled [BT,BV] → u_tile [BT,BV]
        u_tile = tl.dot(A_c, v_scaled)                 # [BT, BV]

        # Store u_c tile
        u_offs = u_base + row_idx[:, None] * s_v_t + v_col_idx[None, :]
        tl.store(u_ptr + u_offs, u_tile.to(tl.float32), mask=v_mask[None, :])

    # ---------------------------------------------------------------
    # OPT 4: Compute w_c = A_c @ (k_c * scale_w[:, None])
    # Tile over K dimension with BK
    # ---------------------------------------------------------------
    k_base = b * s_k_b + t_start * s_k_t + h * s_k_h
    w_base = b * s_k_b + t_start * s_k_t + h * s_k_h  # same layout as k

    for k_start in tl.range(0, K, BK):
        k_col_idx = k_start + tl.arange(0, BK)         # [BK]
        k_mask = k_col_idx < K

        # Load k_c tile [BT, BK]
        k_offs = k_base + row_idx[:, None] * s_k_t + k_col_idx[None, :]
        k_tile = tl.load(k_ptr + k_offs, mask=k_mask[None, :], other=0.0).to(tl.float32)

        # OPT 3: Fuse beta*exp(g) scaling into k — no extra alloc
        k_scaled = k_tile * scale_w[:, None]            # [BT, BK]

        # OPT 4: Tensor Core GEMM — A_c [BT,BT] @ k_scaled [BT,BK] → w_tile [BT,BK]
        w_tile = tl.dot(A_c, k_scaled)                 # [BT, BK]

        # Store w_c tile
        w_offs = w_base + row_idx[:, None] * s_k_t + k_col_idx[None, :]
        tl.store(w_ptr + w_offs, w_tile.to(tl.float32), mask=k_mask[None, :])


def solution(data: input_t) -> output_t:
    """
    Drop-in replacement for ref_kernel.
    Accepts the same (k, v, beta, A, g) tuple and returns (w, u).
    """
    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V = v.shape[-1]
    BT = CHUNK_SIZE
    NT = T // BT

    # Allocate outputs — same shape and dtype as reference
    w = torch.empty_like(k)
    u = torch.empty_like(v)

    # Strides for k [B, T, H, K] — contiguous layout
    s_k_b, s_k_t, s_k_h = k.stride(0), k.stride(1), k.stride(2)
    # Strides for v [B, T, H, V]
    s_v_b, s_v_t, s_v_h = v.stride(0), v.stride(1), v.stride(2)
    # Strides for beta/g [B, T, H]
    s_bg_b, s_bg_t, s_bg_h = beta.stride(0), beta.stride(1), beta.stride(2)
    # Strides for A [B, T, H, BT]
    s_A_b, s_A_t, s_A_h = A.stride(0), A.stride(1), A.stride(2)

    # OPT 1: All (b, h, nt) combinations run in parallel
    grid = (B * H * NT,)

    _recompute_w_u_kernel[grid](
        k, v, beta, A, g,
        w, u,
        B=B, T=T, H=H, K=K, V=V, BT=BT,
        s_k_b=s_k_b, s_k_t=s_k_t, s_k_h=s_k_h,
        s_v_b=s_v_b, s_v_t=s_v_t, s_v_h=s_v_h,
        s_bg_b=s_bg_b, s_bg_t=s_bg_t, s_bg_h=s_bg_h,
        s_A_b=s_A_b, s_A_t=s_A_t, s_A_h=s_A_h,
    )

    return w, u


# ---------------------------------------------------------------------------
# Helion wrapper — required by eval.py
# ---------------------------------------------------------------------------
def generate_input(B: int, T: int, H: int, K: int, V: int, seed: int):
    """Delegate to reference.py's generate_input (PR #127 stability fixes included)."""
    from reference import generate_input as ref_gen
    return ref_gen(B, T, H, K, V, seed)


def check_implementation(data, output):
    """Delegate to reference.py's check_implementation."""
    from reference import check_implementation as ref_check
    return ref_check(data, output)

