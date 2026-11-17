"""Triton implicit-GEMM weight gradient for small-Cin convolutions (ResNet stem: 3->64, 7x7, stride 2, pad 3).

dW[co, ci, kh, kw] = sum_{n,ho,wo} gy[n, co, ho, wo] * x[n, ci, ho*s - p + kh, wo*s - p + kw]
Inputs are channels_last (NHWC memory) bf16; accumulation is fp32 (split-K over pixels, atomic reduction).
Same math as the convolution weight gradient, without materializing the unfolded patch matrix.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _stem_wgrad_kernel(x_ptr, gy_ptr, dw_ptr,
                       N, H, W, Ho, Wo, P,
                       CIN: tl.constexpr, COUT: tl.constexpr, KS: tl.constexpr,
                       STRIDE: tl.constexpr, PAD: tl.constexpr,
                       K: tl.constexpr, BLOCK_K: tl.constexpr, DW_LD: tl.constexpr,
                       BLOCK_P: tl.constexpr, ROWS_PER_PROG: tl.constexpr):
    pid = tl.program_id(0)
    kid = tl.program_id(1)
    offs_co = tl.arange(0, COUT)
    offs_k = kid * BLOCK_K + tl.arange(0, BLOCK_K)
    # k -> (ci, kh, kw) in PyTorch weight order (ci major, then kh, kw)
    ci = offs_k // (KS * KS)
    kh = (offs_k // KS) % KS
    kw = offs_k % KS
    k_ok = offs_k < K
    acc = tl.zeros((COUT, BLOCK_K), dtype=tl.float32)
    start = pid * ROWS_PER_PROG
    for r in range(0, ROWS_PER_PROG, BLOCK_P):
        offs_p = start + r + tl.arange(0, BLOCK_P)
        p_ok = offs_p < P
        n = offs_p // (Ho * Wo)
        rem = offs_p % (Ho * Wo)
        ho = rem // Wo
        wo = rem % Wo
        # gy tile [BLOCK_P, COUT]: NHWC -> contiguous over co
        gy = tl.load(gy_ptr + offs_p[:, None] * COUT + offs_co[None, :], mask=p_ok[:, None], other=0.0)
        # x patch tile [BLOCK_P, BLOCK_K]
        h = ho[:, None] * STRIDE - PAD + kh[None, :]
        w = wo[:, None] * STRIDE - PAD + kw[None, :]
        ok = p_ok[:, None] & k_ok[None, :] & (h >= 0) & (h < H) & (w >= 0) & (w < W)
        xv = tl.load(x_ptr + ((n[:, None] * H + h) * W + w) * CIN + ci[None, :], mask=ok, other=0.0)
        acc += tl.dot(tl.trans(gy), xv)
    out = dw_ptr + offs_co[:, None] * DW_LD + offs_k[None, :]
    tl.atomic_add(out, acc, mask=k_ok[None, :], sem='relaxed')


def stem_wgrad(x, gy, weight_shape, stride, padding, rows_per_prog=4096, block_p=32, block_k=64, num_warps=4, num_stages=1):
    """x: (N,Cin,H,W) channels_last; gy: (N,Cout,Ho,Wo) channels_last. Returns dW like weight_shape (fp32 accum, cast to gy dtype)."""
    cout, cin, ks, ks2 = weight_shape
    assert ks == ks2 and x.shape[1] == cin and gy.shape[1] == cout
    N, _, H, W = x.shape
    Ho, Wo = gy.shape[2], gy.shape[3]
    x = x.contiguous(memory_format=torch.channels_last)
    gy = gy.contiguous(memory_format=torch.channels_last)
    K = cin * ks * ks
    BLOCK_K = block_k or triton.next_power_of_2(K)
    KB = triton.cdiv(K, BLOCK_K)
    DW_LD = KB * BLOCK_K
    P = N * Ho * Wo
    dw = torch.zeros((cout, DW_LD), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(P, rows_per_prog), KB)
    _stem_wgrad_kernel[grid](x, gy, dw, N, H, W, Ho, Wo, P,
                             CIN=cin, COUT=cout, KS=ks, STRIDE=stride, PAD=padding,
                             K=K, BLOCK_K=BLOCK_K, DW_LD=DW_LD, BLOCK_P=block_p, ROWS_PER_PROG=rows_per_prog,
                             num_warps=num_warps, num_stages=num_stages)
    return dw[:, :K].reshape(cout, cin, ks, ks).to(gy.dtype)
