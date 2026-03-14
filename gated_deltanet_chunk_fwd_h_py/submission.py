"""
gated_deltanet_chunk_fwd_h — Pure PyTorch submission
=====================================================
Implements the sequential h_state recurrence across NT chunks using
batched PyTorch tensor operations. No Triton/Helion JIT — avoids all
compilation overhead on the remote runner.

For each (b, h) pair, starting with h_state = zeros(K, V):
  For each chunk c = 0, 1, ..., NT-1:
    1. Store: h_out[b, c, h] = h_state
    2. Compute: v_new = u - w @ h_state
    3. Gate: v_gated[t] = v_new[t] * exp(g[last_t] - g[t])
    4. Decay: h_state = h_state * exp(g[last_t])
    5. Update: h_state = h_state + k^T @ v_gated

The Python loop over NT is unavoidable (sequential dependency), but all
(b, h) pairs and all (K, V) dimensions are processed in parallel.
"""

import torch
from task import input_t, output_t

CHUNK_SIZE = 64


def custom_kernel(data: input_t) -> output_t:
    k, w, u, g = data
    B, T, H, K = k.shape
    V  = u.shape[-1]
    C  = CHUNK_SIZE
    NT = T // C

    # Work in float32 throughout
    k = k.float()
    w = w.float()
    u = u.float()
    g = g.float()

    h_out = torch.empty(B, NT, H, K, V, dtype=torch.float32, device=k.device)
    v_new = torch.empty(B, T,  H, V,    dtype=u.dtype,        device=k.device)

    # h_state: [B, H, K, V] — all (b, h) pairs in parallel
    h_state = torch.zeros(B, H, K, V, dtype=torch.float32, device=k.device)

    for c in range(NT):
        t0 = c * C
        t1 = t0 + C

        # Store current h_state
        h_out[:, c, :, :, :] = h_state   # [B, H, K, V]

        # Slice this chunk: [B, C, H, *]
        k_c = k[:, t0:t1, :, :]   # [B, C, H, K]
        w_c = w[:, t0:t1, :, :]   # [B, C, H, K]
        u_c = u[:, t0:t1, :, :]   # [B, C, H, V]
        g_c = g[:, t0:t1, :  ]    # [B, C, H]

        g_last = g_c[:, -1, :].unsqueeze(1)   # [B, 1, H]

        # v_new_c = u_c - w_c @ h_state
        # w_c: [B, C, H, K], h_state: [B, H, K, V]
        # w_c @ h: need [B, H, C, K] @ [B, H, K, V] -> [B, H, C, V]
        w_t = w_c.permute(0, 2, 1, 3)   # [B, H, C, K]
        u_t = u_c.permute(0, 2, 1, 3)   # [B, H, C, V]
        k_t = k_c.permute(0, 2, 1, 3)   # [B, H, C, K]
        g_t = g_c.permute(0, 2, 1)      # [B, H, C]

        wh = w_t @ h_state              # [B, H, C, V]
        vn = u_t - wh                   # [B, H, C, V]

        # gate = exp(g_last - g_c): [B, H, C]
        g_last_t = g_last.permute(0, 2, 1)   # [B, H, 1]
        gate = torch.exp(g_last_t - g_t)     # [B, H, C]
        v_gated = vn * gate.unsqueeze(-1)    # [B, H, C, V]

        # Write v_new back: [B, C, H, V]
        v_new[:, t0:t1, :, :] = vn.permute(0, 2, 1, 3).to(v_new.dtype)

        # Decay and update h_state
        decay = torch.exp(g_last_t.squeeze(-1))   # [B, H]
        h_state = h_state * decay.unsqueeze(-1).unsqueeze(-1)   # [B, H, K, V]
        h_state = h_state + k_t.transpose(-1, -2) @ v_gated     # [B, H, K, V]

    return h_out, v_new

