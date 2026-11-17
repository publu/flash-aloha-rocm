"""Triton implicit-GEMM convolution weight gradient, NHWC bf16, fp32 accumulation, split-K over output pixels.

dW[co, kh, kw, ci] = sum_{n,ho,wo} gy[n, ho, wo, co] * x[n, ho*s - p + kh, wo*s - p + kw, ci]
Reduction index k = (kh*KS + kw)*CIN + ci, i.e. the channels_last weight layout, so loads of x are contiguous over ci.
Grid: (pixel chunks, Cout blocks, K blocks); partial sums are reduced with fp32 atomics.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _wgrad_kernel(x_ptr, gy_ptr, dw_ptr, H, W, Ho, Wo, P,
                  CIN: tl.constexpr, COUT: tl.constexpr, KS: tl.constexpr, STRIDE: tl.constexpr, PAD: tl.constexpr,
                  K: tl.constexpr, BLOCK_CO: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_P: tl.constexpr,
                  ROWS_PER_PROG: tl.constexpr):
    pid = tl.program_id(0)
    cob = tl.program_id(1)
    kb = tl.program_id(2)
    offs_co = cob * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
    ci = offs_k % CIN
    tap = offs_k // CIN
    kh = tap // KS
    kw = tap % KS
    k_ok = offs_k < K
    co_ok = offs_co < COUT
    acc = tl.zeros((BLOCK_CO, BLOCK_K), dtype=tl.float32)
    start = pid * ROWS_PER_PROG
    for r in range(0, ROWS_PER_PROG, BLOCK_P):
        offs_p = start + r + tl.arange(0, BLOCK_P)
        p_ok = offs_p < P
        n = offs_p // (Ho * Wo)
        rem = offs_p % (Ho * Wo)
        ho = rem // Wo
        wo = rem % Wo
        gy = tl.load(gy_ptr + offs_p[:, None] * COUT + offs_co[None, :], mask=p_ok[:, None] & co_ok[None, :], other=0.0)
        h = ho[:, None] * STRIDE - PAD + kh[None, :]
        w = wo[:, None] * STRIDE - PAD + kw[None, :]
        ok = p_ok[:, None] & k_ok[None, :] & (h >= 0) & (h < H) & (w >= 0) & (w < W)
        xv = tl.load(x_ptr + ((n[:, None] * H + h) * W + w) * CIN + ci[None, :], mask=ok, other=0.0)
        acc += tl.dot(tl.trans(gy), xv)
    out = dw_ptr + offs_co[:, None] * K + offs_k[None, :]
    tl.atomic_add(out, acc, mask=co_ok[:, None] & k_ok[None, :], sem='relaxed')


def conv_wgrad(x, gy, weight_shape, stride, padding, *, rows_per_prog=2048, block_p=32, block_co=64, block_k=64,
               num_warps=4, num_stages=1):
    """x (N,Cin,H,W), gy (N,Cout,Ho,Wo), any strides (made channels_last). Returns dW (Cout,Cin,KS,KS) fp32 with
    channels_last strides."""
    cout, cin, ks, ks2 = weight_shape
    assert ks == ks2
    N, _, H, W = x.shape
    Ho, Wo = gy.shape[2], gy.shape[3]
    x = x.contiguous(memory_format=torch.channels_last)
    gy = gy.contiguous(memory_format=torch.channels_last)
    K = ks * ks * cin
    P = N * Ho * Wo
    dw = torch.zeros((cout, K), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(P, rows_per_prog), triton.cdiv(cout, block_co), triton.cdiv(K, block_k))
    _wgrad_kernel[grid](x, gy, dw, H, W, Ho, Wo, P, CIN=cin, COUT=cout, KS=ks, STRIDE=stride, PAD=padding, K=K,
                        BLOCK_CO=block_co, BLOCK_K=block_k, BLOCK_P=block_p, ROWS_PER_PROG=rows_per_prog,
                        num_warps=num_warps, num_stages=num_stages)
    # (cout, kh, kw, ci) memory == channels_last weight
    return dw.view(cout, ks, ks, cin).permute(0, 3, 1, 2)
