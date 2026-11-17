"""Triton attention for short sequences (ACT: L~201, head dim 64), bf16 in/out, fp32 softmax/accumulation, with
attention dropout. Same math as F.scaled_dot_product_attention(q, k, v, dropout_p=p) (no mask, not causal):
    P = softmax(q k^T * scale),  O = (P * M / (1-p)) v,   M ~ Bernoulli(1-p) from Philox(seed, (bh*L + i)*L + j)
The backward regenerates M from the same seed/offsets (nothing stored), and uses the saved row logsumexp.
Inputs may be strided (B, H, L, D) views with a contiguous last dim (as nn.MultiheadAttention passes them).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd(Q, K, V, O, LSE, seed, sm_scale, p_drop, keep_scale, L, H,
         sqb, sqh, sql, skb, skh, skl, svb, svh, svl, sob, soh, sol,
         D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, DROPOUT: tl.constexpr):
    mb = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_m = mb * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    mm = offs_m < L
    q = tl.load(Q + b * sqb + h * sqh + offs_m[:, None] * sql + offs_d[None, :], mask=mm[:, None], other=0.0)
    m_i = tl.full((BM,), float('-inf'), tl.float32)
    l_i = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    qk_scale = sm_scale * 1.4426950408889634  # exp2
    for n0 in range(0, L, BN):
        kn = n0 + offs_n
        nm = kn < L
        k = tl.load(K + b * skb + h * skh + kn[:, None] * skl + offs_d[None, :], mask=nm[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        s = tl.where(nm[None, :], s, float('-inf'))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp2(s - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        if DROPOUT:
            rnd = tl.rand(seed, (bh * L + offs_m[:, None]) * L + kn[None, :])
            p = tl.where(rnd >= p_drop, p * keep_scale, 0.0)
        v = tl.load(V + b * svb + h * svh + kn[:, None] * svl + offs_d[None, :], mask=nm[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(O + b * sob + h * soh + offs_m[:, None] * sol + offs_d[None, :], acc.to(O.dtype.element_ty),
             mask=mm[:, None])
    tl.store(LSE + bh * L + offs_m, m_i + tl.log2(l_i), mask=mm)  # base-2 logsumexp of scaled scores


@triton.jit
def _bwd_dkdv(Q, K, V, DO, LSE, DELTA, DK, DV, seed, sm_scale, p_drop, keep_scale, L, H,
              sqb, sqh, sql, skb, skh, skl, svb, svh, svl, sdb, sdh, sdl,
              D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, DROPOUT: tl.constexpr):
    nb = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_n = nb * BN + tl.arange(0, BN)
    offs_m0 = tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    nm = offs_n < L
    k = tl.load(K + b * skb + h * skh + offs_n[:, None] * skl + offs_d[None, :], mask=nm[:, None], other=0.0)
    v = tl.load(V + b * svb + h * svh + offs_n[:, None] * svl + offs_d[None, :], mask=nm[:, None], other=0.0)
    dk = tl.zeros((BN, D), tl.float32)
    dv = tl.zeros((BN, D), tl.float32)
    qk_scale = sm_scale * 1.4426950408889634
    for m0 in range(0, L, BM):
        offs_m = m0 + offs_m0
        mm = offs_m < L
        q = tl.load(Q + b * sqb + h * sqh + offs_m[:, None] * sql + offs_d[None, :], mask=mm[:, None], other=0.0)
        do = tl.load(DO + b * sdb + h * sdh + offs_m[:, None] * sdl + offs_d[None, :], mask=mm[:, None], other=0.0)
        lse = tl.load(LSE + bh * L + offs_m, mask=mm, other=0.0)
        delta = tl.load(DELTA + bh * L + offs_m, mask=mm, other=0.0)
        s = tl.dot(q, tl.trans(k)) * qk_scale                      # (BM, BN)
        p = tl.exp2(s - lse[:, None])
        p = tl.where(mm[:, None] & nm[None, :], p, 0.0)
        dp = tl.dot(do, tl.trans(v))                               # dO V^T (BM, BN)
        if DROPOUT:
            rnd = tl.rand(seed, (bh * L + offs_m[:, None]) * L + offs_n[None, :])
            keep = rnd >= p_drop
            pz = tl.where(keep, p * keep_scale, 0.0)
            dp = tl.where(keep, dp * keep_scale, 0.0)
        else:
            pz = p
        dv += tl.dot(tl.trans(pz.to(do.dtype)), do)
        ds = p * (dp - delta[:, None])
        dk += tl.dot(tl.trans(ds.to(q.dtype)), q)
    dk = dk * sm_scale
    tl.store(DK + bh * L * D + offs_n[:, None] * D + offs_d[None, :], dk.to(DK.dtype.element_ty), mask=nm[:, None])
    tl.store(DV + bh * L * D + offs_n[:, None] * D + offs_d[None, :], dv.to(DV.dtype.element_ty), mask=nm[:, None])


@triton.jit
def _bwd_dq(Q, K, V, DO, LSE, DELTA, DQ, seed, sm_scale, p_drop, keep_scale, L, H,
            sqb, sqh, sql, skb, skh, skl, svb, svh, svl, sdb, sdh, sdl,
            D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, DROPOUT: tl.constexpr):
    mb = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_m = mb * BM + tl.arange(0, BM)
    offs_n0 = tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    mm = offs_m < L
    q = tl.load(Q + b * sqb + h * sqh + offs_m[:, None] * sql + offs_d[None, :], mask=mm[:, None], other=0.0)
    do = tl.load(DO + b * sdb + h * sdh + offs_m[:, None] * sdl + offs_d[None, :], mask=mm[:, None], other=0.0)
    lse = tl.load(LSE + bh * L + offs_m, mask=mm, other=0.0)
    delta = tl.load(DELTA + bh * L + offs_m, mask=mm, other=0.0)
    dq = tl.zeros((BM, D), tl.float32)
    qk_scale = sm_scale * 1.4426950408889634
    for n0 in range(0, L, BN):
        offs_n = n0 + offs_n0
        nm = offs_n < L
        k = tl.load(K + b * skb + h * skh + offs_n[:, None] * skl + offs_d[None, :], mask=nm[:, None], other=0.0)
        v = tl.load(V + b * svb + h * svh + offs_n[:, None] * svl + offs_d[None, :], mask=nm[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * qk_scale
        p = tl.exp2(s - lse[:, None])
        p = tl.where(mm[:, None] & nm[None, :], p, 0.0)
        dp = tl.dot(do, tl.trans(v))
        if DROPOUT:
            rnd = tl.rand(seed, (bh * L + offs_m[:, None]) * L + offs_n[None, :])
            dp = tl.where(rnd >= p_drop, dp * keep_scale, 0.0)
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(k.dtype), k)
    dq = dq * sm_scale
    tl.store(DQ + bh * L * D + offs_m[:, None] * D + offs_d[None, :], dq.to(DQ.dtype.element_ty), mask=mm[:, None])


def _s(t):
    return t.stride(0), t.stride(1), t.stride(2)


CFG = dict(BM=64, BN=64, num_warps=4, num_stages=1)


def attn_fwd(q, k, v, p_drop, seed, scale=None, cfg=None):
    c = dict(CFG, **(cfg or {}))
    B, H, L, D = q.shape
    sm_scale = scale if scale is not None else D ** -0.5
    o = torch.empty((L, B, H, D), device=q.device, dtype=q.dtype).permute(1, 2, 0, 3)  # free permute back in MHA
    lse = torch.empty((B * H, L), device=q.device, dtype=torch.float32)
    keep = 1.0 / (1.0 - p_drop) if p_drop > 0 else 1.0
    grid = (triton.cdiv(L, c['BM']), B * H)
    _fwd[grid](q, k, v, o, lse, seed, sm_scale, p_drop, keep, L, H, *_s(q), *_s(k), *_s(v), *_s(o),
               D=D, BM=c['BM'], BN=c['BN'], DROPOUT=p_drop > 0, num_warps=c['num_warps'], num_stages=c['num_stages'])
    return o, lse


def attn_bwd(do, q, k, v, o, lse, p_drop, seed, scale=None, cfg=None):
    c = dict(CFG, **(cfg or {}))
    B, H, L, D = q.shape
    sm_scale = scale if scale is not None else D ** -0.5
    keep = 1.0 / (1.0 - p_drop) if p_drop > 0 else 1.0
    delta = (do.float() * o.float()).sum(-1).reshape(B * H, L)
    dq = torch.empty((B, H, L, D), device=q.device, dtype=q.dtype)
    dk = torch.empty_like(dq)
    dv = torch.empty_like(dq)
    args = (seed, sm_scale, p_drop, keep, L, H, q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1),
            k.stride(2), v.stride(0), v.stride(1), v.stride(2), do.stride(0), do.stride(1), do.stride(2))
    kw = dict(D=D, BM=c['BM'], BN=c['BN'], DROPOUT=p_drop > 0, num_warps=c['num_warps'], num_stages=c['num_stages'])
    _bwd_dkdv[(triton.cdiv(L, c['BN']), B * H)](q, k, v, do, lse, delta, dk, dv, *args, **kw)
    _bwd_dq[(triton.cdiv(L, c['BM']), B * H)](q, k, v, do, lse, delta, dq, *args, **kw)
    return dq, dk, dv


@triton.jit
def _mask_kernel(M, seed, p_drop, L, BLOCK: tl.constexpr):
    """Writes the keep-mask the kernels use, for testing only."""
    bh = tl.program_id(1)
    i = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    rnd = tl.rand(seed, (bh * L + i) * L + j)
    tl.store(M + (bh * L + i) * L + j, (rnd >= p_drop).to(tl.int8), mask=j < L)


def dropout_mask(B, H, L, p_drop, seed, device='cuda'):
    m = torch.empty((B * H, L, L), device=device, dtype=torch.int8)
    _mask_kernel[(L, B * H)](m, seed, p_drop, L, BLOCK=triton.next_power_of_2(L))
    return m.view(B, H, L, L).bool()
