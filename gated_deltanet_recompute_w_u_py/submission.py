"""
gated_deltanet_recompute_w_u — Optimised Triton Submission
===========================================================
Computes for each chunk independently (fully parallel — no inter-chunk dependency):
    u_c = A_c @ (v_c * beta_c[:, None])
    w_c = A_c @ (k_c * (beta_c * exp(g_c))[:, None])

Key optimisation: A_c is loaded ONCE and reused for both the u and w GEMMs.
The reference loads A_c twice (once per torch.matmul call).

All chunks are independent -> grid = (B * H * NT,).

Compilation note: K and V are NOT tl.constexpr — one PTX binary covers all
shapes (K=64, K=100->128, K=128). BK_ and BV_ are next_pow2(K/V) and ARE
constexpr so tl.dot() gets compile-time tile sizes. At most 2 compilations
(BK_=64 and BK_=128) instead of 7+.
"""

import math
import torch
import triton
import triton.language as tl

from task import input_t, output_t

BT = 64   # chunk size (fixed by the problem)


@triton.jit
def _recompute_w_u_kernel(
    k_ptr, v_ptr, beta_ptr, A_ptr, g_ptr,
    w_ptr, u_ptr,
    # strides for k [B, T, H, K]
    k_sb, k_st, k_sh, k_sk,
    # strides for v [B, T, H, V]
    v_sb, v_st, v_sh, v_sv,
    # strides for beta [B, T, H]
    b_sb, b_st, b_sh,
    # strides for A [B, T, H, BT]  (BT=64 always)
    A_sb, A_st, A_sh, A_sbt,
    # strides for g [B, T, H]
    g_sb, g_st, g_sh,
    # strides for w [B, T, H, K]
    w_sb, w_st, w_sh, w_sk,
    # strides for u [B, T, H, V]
    u_sb, u_st, u_sh, u_sv,
    B: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
    NT: tl.constexpr, C: tl.constexpr,
    BK_: tl.constexpr, BV_: tl.constexpr,
    K_real: tl.constexpr, V_real: tl.constexpr,
):
    # Grid: (B * H * NT,)
    pid = tl.program_id(0)
    nt  = pid % NT
    bh  = pid // NT
    b   = bh // H
    h   = bh  % H

    t_start = nt * C
    c_range = tl.arange(0, C)      # [C]  — chunk timestep indices
    k_range = tl.arange(0, BK_)   # [BK_] — key/query dimension
    v_range = tl.arange(0, BV_)   # [BV_] — value dimension

    k_mask = k_range < K_real
    v_mask = v_range < V_real

    # Base pointers for this (b, h, chunk)
    t_abs = t_start + c_range   # absolute time indices for this chunk

    k_base = k_ptr + b * k_sb + h * k_sh
    v_base = v_ptr + b * v_sb + h * v_sh
    b_base = beta_ptr + b * b_sb + h * b_sh
    A_base = A_ptr + b * A_sb + h * A_sh
    g_base = g_ptr + b * g_sb + h * g_sh
    w_base = w_ptr + b * w_sb + h * w_sh
    u_base = u_ptr + b * u_sb + h * u_sh

    # Load A_c [C, C] — the WY matrix for this chunk (ONCE, reused for both GEMMs)
    # A is stored as [B, T, H, BT] where T indexes the row and BT indexes the column
    A_off = t_abs[:, None] * A_st + c_range[None, :] * A_sbt   # [C, C]
    A_c = tl.load(A_base + A_off).to(tl.float32)   # [C, C]

    # Load beta_c [C] and g_c [C]
    b_off  = t_abs * b_st
    g_off  = t_abs * g_st
    beta_c = tl.load(b_base + b_off).to(tl.float32)   # [C]
    g_c    = tl.load(g_base + g_off).to(tl.float32)   # [C]

    # scale_u[t] = beta_c[t]
    # scale_w[t] = beta_c[t] * exp(g_c[t])
    scale_u = beta_c                          # [C]
    scale_w = beta_c * tl.exp(g_c)           # [C]

    # Load k_c [C, K] and v_c [C, V]
    k_off = t_abs[:, None] * k_st + k_range[None, :] * k_sk   # [C, BK_]
    v_off = t_abs[:, None] * v_st + v_range[None, :] * v_sv   # [C, BV_]

    k_c = tl.load(k_base + k_off,
                  mask=(c_range < C)[:, None] & k_mask[None, :],
                  other=0.0).to(tl.float32)   # [C, BK_]
    v_c = tl.load(v_base + v_off,
                  mask=(c_range < C)[:, None] & v_mask[None, :],
                  other=0.0).to(tl.float32)   # [C, BV_]

    # Scale: k_scaled[t] = k_c[t] * scale_w[t]
    #        v_scaled[t] = v_c[t] * scale_u[t]
    k_scaled = k_c * scale_w[:, None]   # [C, BK_]
    v_scaled = v_c * scale_u[:, None]   # [C, BV_]

    # w_c = A_c @ k_scaled  [C, BK_]
    # u_c = A_c @ v_scaled  [C, BV_]
    w_c = tl.dot(A_c, k_scaled)   # [C, BK_]
    u_c = tl.dot(A_c, v_scaled)   # [C, BV_]

    # Store w_c and u_c
    w_off = t_abs[:, None] * w_st + k_range[None, :] * w_sk
    u_off = t_abs[:, None] * u_st + v_range[None, :] * u_sv

    tl.store(w_base + w_off, w_c.to(k_ptr.dtype.element_ty),
             mask=(c_range < C)[:, None] & k_mask[None, :])
    tl.store(u_base + u_off, u_c.to(v_ptr.dtype.element_ty),
             mask=(c_range < C)[:, None] & v_mask[None, :])


def _next_pow2(n: int) -> int:
    return 1 << math.ceil(math.log2(max(n, 1)))


def custom_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V  = v.shape[-1]
    NT = T // BT
    C  = BT

    BK_ = max(64, _next_pow2(K))
    BV_ = max(64, _next_pow2(V))

    w_out = torch.empty_like(k)
    u_out = torch.empty_like(v)

    grid = (B * H * NT,)

    _recompute_w_u_kernel[grid](
        k, v, beta, A, g, w_out, u_out,
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        beta.stride(0), beta.stride(1), beta.stride(2),
        A.stride(0), A.stride(1), A.stride(2), A.stride(3),
        g.stride(0), g.stride(1), g.stride(2),
        w_out.stride(0), w_out.stride(1), w_out.stride(2), w_out.stride(3),
        u_out.stride(0), u_out.stride(1), u_out.stride(2), u_out.stride(3),
        B=B, T=T, H=H, NT=NT, C=C,
        BK_=BK_, BV_=BV_,
        K_real=K, V_real=V,
        num_warps=4,
        num_stages=2,
    )
    return w_out, u_out

