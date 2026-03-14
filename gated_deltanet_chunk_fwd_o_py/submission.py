"""
gated_deltanet_chunk_fwd_o — Pure PyTorch submission
=====================================================
Uses the exact same formula as the fixed ref_kernel (with causal masking
before exp to avoid NaN), but processes all NT chunks in parallel with
batched tensor operations instead of a Python for loop.

No Triton/Helion JIT — avoids all compilation overhead on the remote runner.
"""

import torch
from task import input_t, output_t

CHUNK_SIZE = 64


def _chunk_fwd_o(q, k, v_new, h, g) -> torch.Tensor:
    B, T, H, K = q.shape
    V  = v_new.shape[-1]
    C  = CHUNK_SIZE
    NT = T // C
    scale = K ** -0.5

    # Reshape to [B, NT, H, C, *] — all chunks in parallel
    q_c = q.float().reshape(B, NT, C, H, K).permute(0, 1, 3, 2, 4)    # [B,NT,H,C,K]
    k_c = k.float().reshape(B, NT, C, H, K).permute(0, 1, 3, 2, 4)    # [B,NT,H,C,K]
    v_c = v_new.float().reshape(B, NT, C, H, V).permute(0, 1, 3, 2, 4) # [B,NT,H,C,V]
    g_c = g.float().reshape(B, NT, C, H).permute(0, 1, 3, 2)           # [B,NT,H,C]

    # inter-chunk: (q @ h) * exp(g)
    o_inter = (q_c @ h.float()) * torch.exp(g_c).unsqueeze(-1)   # [B,NT,H,C,V]

    # intra-chunk: causal_mask(q @ k^T * exp(g_i - g_j)) @ v_new
    # Apply causal mask BEFORE exp to avoid NaN (g_diff in upper triangle can be large positive)
    causal = torch.tril(torch.ones(C, C, dtype=torch.bool, device=q.device))
    g_diff = g_c.unsqueeze(-1) - g_c.unsqueeze(-2)                # [B,NT,H,C,C]
    g_diff = torch.where(causal, g_diff, torch.zeros_like(g_diff)) # zero upper tri
    qk = q_c @ k_c.transpose(-1, -2) * torch.exp(g_diff) * causal # [B,NT,H,C,C]
    o_intra = qk @ v_c                                              # [B,NT,H,C,V]

    o = (o_inter + o_intra) * scale
    return o.permute(0, 1, 3, 2, 4).reshape(B, T, H, V).to(q.dtype)


def custom_kernel(data: input_t) -> output_t:
    q, k, v_new, h, g = data
    return _chunk_fwd_o(q, k, v_new, h, g)

