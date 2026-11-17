"""Triton forward 3x3 / stride 1 / pad 1 convolution, NHWC bf16, fp32 accumulation, no padded copies.
y[p, co] = sum_{kh,kw,ci} x[p + (kh-1)*W + (kw-1), ci] * w[co, kh, kw, ci]   (border rows masked per pixel)
Used for the forward pass and for the input gradient (forward conv with flip_transpose(w))."""
import torch
import triton
import triton.language as tl


@triton.jit
def _conv3x3_fwd_kernel(x_ptr, w_ptr, y_ptr, P, H, W,
                        CIN: tl.constexpr, COUT: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_CO: tl.constexpr,
                        BLOCK_CI: tl.constexpr, GROUP_P: tl.constexpr):
    # grouped launch order so consecutive programs share x rows (L2 reuse)
    pid = tl.program_id(0)
    n_co: tl.constexpr = COUT // BLOCK_CO
    n_p = tl.cdiv(P, BLOCK_P)
    group = pid // (GROUP_P * n_co)
    first_p = group * GROUP_P
    gsize = tl.minimum(n_p - first_p, GROUP_P)
    pb = first_p + (pid % (GROUP_P * n_co)) % gsize
    cb = (pid % (GROUP_P * n_co)) // gsize
    p = pb * BLOCK_P + tl.arange(0, BLOCK_P)
    co = cb * BLOCK_CO + tl.arange(0, BLOCK_CO)
    ci = tl.arange(0, BLOCK_CI)
    ok = p < P
    wcol = p % W
    hrow = (p // W) % H
    acc = tl.zeros((BLOCK_P, BLOCK_CO), dtype=tl.float32)
    for kh in tl.static_range(3):
        hh = hrow + kh - 1
        okh = ok & (hh >= 0) & (hh < H)
        for kw in tl.static_range(3):
            ww = wcol + kw - 1
            m = okh & (ww >= 0) & (ww < W)
            src = p + (kh - 1) * W + (kw - 1)
            for c0 in range(0, CIN, BLOCK_CI):
                xv = tl.load(x_ptr + src[:, None] * CIN + (c0 + ci)[None, :], mask=m[:, None], other=0.0)
                wv = tl.load(w_ptr + (co[None, :] * 9 + kh * 3 + kw) * CIN + (c0 + ci)[:, None])
                acc += tl.dot(xv, wv)
    tl.store(y_ptr + p[:, None] * COUT + co[None, :], acc.to(y_ptr.dtype.element_ty), mask=ok[:, None])


def conv3x3_fwd(x, w, *, block_p=128, block_co=64, block_ci=64, group_p=8, num_warps=4, num_stages=2):
    """x (N,Cin,H,W), w (Cout,Cin,3,3); returns y (N,Cout,H,W) channels_last, dtype of x."""
    cout, cin = w.shape[0], w.shape[1]
    assert w.shape[2:] == (3, 3) and cin % block_ci == 0 and cout % block_co == 0
    x = x.contiguous(memory_format=torch.channels_last)
    w = w.contiguous(memory_format=torch.channels_last)
    N, _, H, W = x.shape
    P = N * H * W
    y = torch.empty((N, cout, H, W), device=x.device, dtype=x.dtype, memory_format=torch.channels_last)
    grid = (triton.cdiv(P, block_p) * (cout // block_co),)
    _conv3x3_fwd_kernel[grid](x, w, y, P, H, W, CIN=cin, COUT=cout, BLOCK_P=block_p, BLOCK_CO=block_co,
                              BLOCK_CI=block_ci, GROUP_P=group_p, num_warps=num_warps, num_stages=num_stages)
    return y
