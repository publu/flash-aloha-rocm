"""Weight gradient for stride-1 'same' convolutions (ks odd, pad = ks//2), NHWC bf16, fp32 accumulation.

Shift trick: on the zero-padded grid (N, H+2p, W+2p), flatten rows r. If gy is placed on the same padded grid
(zeros outside the valid HxW), then for tap (kh, kw)
    dW[:, kh, kw, :] = sum_r gy_pg[r, :]^T  x_pad[r + kh*(W+2p) + kw, :]
i.e. every tap is a plain contiguous GEMM (Cout x R) @ (R x Cin) at a constant row offset. Same math as the
convolution weight gradient; the zero rows contribute nothing. No per-element index math in the kernel.

Also: 1x1 stride-s convolutions (ResNet downsample) = one GEMM on the subsampled input.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _shift_wgrad_kernel(x_ptr, gy_ptr, dw_ptr, R, WP, RA,
                        CIN: tl.constexpr, COUT: tl.constexpr, KS: tl.constexpr,
                        BLOCK_CO: tl.constexpr, BLOCK_CI: tl.constexpr, BLOCK_P: tl.constexpr,
                        ROWS_PER_PROG: tl.constexpr):
    pid = tl.program_id(0)
    tap = tl.program_id(1)
    cb = tl.program_id(2)
    n_ci_blocks: tl.constexpr = CIN // BLOCK_CI
    cob = cb // n_ci_blocks
    cib = cb % n_ci_blocks
    kh = tap // KS
    kw = tap % KS
    off = kh * WP + kw
    offs_co = cob * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci = cib * BLOCK_CI + tl.arange(0, BLOCK_CI)
    acc = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    start = pid * ROWS_PER_PROG
    for r in range(0, ROWS_PER_PROG, BLOCK_P):
        rows = start + r + tl.arange(0, BLOCK_P)
        ok = rows < R
        g = tl.load(gy_ptr + rows[:, None] * COUT + offs_co[None, :], mask=ok[:, None], other=0.0)
        xr = rows + off
        xv = tl.load(x_ptr + xr[:, None] * CIN + offs_ci[None, :], mask=(xr < RA)[:, None], other=0.0)
        acc += tl.dot(tl.trans(g), xv)
    out = dw_ptr + (offs_co[:, None] * (KS * KS) + tap) * CIN + offs_ci[None, :]
    tl.atomic_add(out, acc, sem='relaxed')


def _nhwc(t):
    return t.permute(0, 2, 3, 1)  # view of a channels_last NCHW tensor as contiguous NHWC


def shift_wgrad(x, gy, weight_shape, padding, *, rows_per_prog=4096, block_p=64, block_co=64, block_ci=64,
                num_warps=4, num_stages=2):
    cout, cin, ks, _ = weight_shape
    assert ks % 2 == 1 and padding == ks // 2 and cin % block_ci == 0 and cout % block_co == 0
    p = padding
    x = x.contiguous(memory_format=torch.channels_last)
    gy = gy.contiguous(memory_format=torch.channels_last)
    N, _, H, W = x.shape
    HP, WP = H + 2 * p, W + 2 * p
    xp = F.pad(_nhwc(x), (0, 0, p, p, p, p))                   # (N, HP, WP, Cin), zero border
    # gy on the padded grid, anchored at the top-left: gy_pg[n, h, w] = gy[n, h, w] for h<H, w<W, else 0
    gpg = F.pad(_nhwc(gy), (0, 0, 0, 2 * p, 0, 2 * p))         # (N, HP, WP, Cout)
    R = N * HP * WP
    dw = torch.zeros((cout, ks * ks, cin), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(R, rows_per_prog), ks * ks, (cout // block_co) * (cin // block_ci))
    _shift_wgrad_kernel[grid](xp, gpg, dw, R, WP, R, CIN=cin, COUT=cout, KS=ks, BLOCK_CO=block_co,
                              BLOCK_CI=block_ci, BLOCK_P=block_p, ROWS_PER_PROG=rows_per_prog,
                              num_warps=num_warps, num_stages=num_stages)
    return dw.view(cout, ks, ks, cin).permute(0, 3, 1, 2)      # channels_last strides


def shift_wgrad_bmm(x, gy, weight_shape, padding):
    """Same shift trick on hipBLASLt: one GEMM per tap on offset row views."""
    cout, cin, ks, _ = weight_shape
    p = padding
    x = x.contiguous(memory_format=torch.channels_last)
    gy = gy.contiguous(memory_format=torch.channels_last)
    N, _, H, W = x.shape
    HP, WP = H + 2 * p, W + 2 * p
    R = N * HP * WP
    xp = F.pad(_nhwc(x), (0, 0, p, p, p, p, 0, 0)).reshape(R, cin)
    xp = torch.cat([xp, xp.new_zeros(ks * WP + ks, cin)])     # tail so every shifted view stays in bounds
    gpg = F.pad(_nhwc(gy), (0, 0, 0, 2 * p, 0, 2 * p)).reshape(R, cout)
    offs = [kh * WP + kw for kh in range(ks) for kw in range(ks)]
    out = torch.stack([gpg.t() @ xp[o:o + R] for o in offs], 1)  # (cout, ks*ks, cin), one GEMM per tap
    return out.view(cout, ks, ks, cin).permute(0, 3, 1, 2).float()


def pointwise_wgrad(x, gy, weight_shape, stride):
    """1x1 conv, any stride, pad 0: dW = gy^T @ x[:, :, ::s, ::s] as one GEMM."""
    cout, cin, _, _ = weight_shape
    xs = _nhwc(x.contiguous(memory_format=torch.channels_last))[:, ::stride, ::stride, :]
    Ho, Wo = gy.shape[2], gy.shape[3]
    xs = xs[:, :Ho, :Wo, :].reshape(-1, cin)
    g = _nhwc(gy.contiguous(memory_format=torch.channels_last)).reshape(-1, cout)
    return (g.t() @ xs).view(cout, 1, 1, cin).permute(0, 3, 1, 2)


@triton.jit
def _shift_wgrad_row_kernel(x_ptr, gy_ptr, dw_ptr, R, WP, RA,
                            CIN: tl.constexpr, COUT: tl.constexpr, KS: tl.constexpr,
                            BLOCK_CO: tl.constexpr, BLOCK_CI: tl.constexpr, BLOCK_P: tl.constexpr,
                            ROWS_PER_PROG: tl.constexpr):
    """Like _shift_wgrad_kernel, but one program computes all KS taps of one kernel row (kh): the gy tile is loaded
    once per row block and reused for the KS kw-shifted x tiles (KS accumulators). Requires KS == 3."""
    pid = tl.program_id(0)
    kh = tl.program_id(1)
    cb = tl.program_id(2)
    n_ci_blocks: tl.constexpr = CIN // BLOCK_CI
    cob = cb // n_ci_blocks
    cib = cb % n_ci_blocks
    offs_co = cob * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci = cib * BLOCK_CI + tl.arange(0, BLOCK_CI)
    a0 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    a1 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    a2 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    start = pid * ROWS_PER_PROG
    base = kh * WP
    for r in range(0, ROWS_PER_PROG, BLOCK_P):
        rows = start + r + tl.arange(0, BLOCK_P)
        ok = rows < R
        g = tl.trans(tl.load(gy_ptr + rows[:, None] * COUT + offs_co[None, :], mask=ok[:, None], other=0.0))
        xr = rows + base
        x0 = tl.load(x_ptr + xr[:, None] * CIN + offs_ci[None, :], mask=(xr < RA)[:, None], other=0.0)
        x1 = tl.load(x_ptr + (xr + 1)[:, None] * CIN + offs_ci[None, :], mask=((xr + 1) < RA)[:, None], other=0.0)
        x2 = tl.load(x_ptr + (xr + 2)[:, None] * CIN + offs_ci[None, :], mask=((xr + 2) < RA)[:, None], other=0.0)
        a0 += tl.dot(g, x0)
        a1 += tl.dot(g, x1)
        a2 += tl.dot(g, x2)
    out = dw_ptr + (offs_co[:, None] * (KS * KS) + kh * KS) * CIN + offs_ci[None, :]
    tl.atomic_add(out, a0, sem='relaxed')
    tl.atomic_add(out + CIN, a1, sem='relaxed')
    tl.atomic_add(out + 2 * CIN, a2, sem='relaxed')


def shift_wgrad_row(x, gy, weight_shape, padding, *, rows_per_prog=8192, block_p=64, block_co=64, block_ci=64,
                    num_warps=4, num_stages=1):
    cout, cin, ks, _ = weight_shape
    assert ks == 3 and padding == 1 and cin % block_ci == 0 and cout % block_co == 0
    p = padding
    x = x.contiguous(memory_format=torch.channels_last)
    gy = gy.contiguous(memory_format=torch.channels_last)
    N, _, H, W = x.shape
    HP, WP = H + 2 * p, W + 2 * p
    xp = F.pad(_nhwc(x), (0, 0, p, p, p, p))
    gpg = F.pad(_nhwc(gy), (0, 0, 0, 2 * p, 0, 2 * p))
    R = N * HP * WP
    dw = torch.zeros((cout, ks * ks, cin), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(R, rows_per_prog), ks, (cout // block_co) * (cin // block_ci))
    _shift_wgrad_row_kernel[grid](xp, gpg, dw, R, WP, R, CIN=cin, COUT=cout, KS=ks, BLOCK_CO=block_co,
                                  BLOCK_CI=block_ci, BLOCK_P=block_p, ROWS_PER_PROG=rows_per_prog,
                                  num_warps=num_warps, num_stages=num_stages)
    return dw.view(cout, ks, ks, cin).permute(0, 3, 1, 2)


@triton.jit
def _wgrad3x3_nopad_kernel(x_ptr, gy_ptr, dw_ptr, P, H, W,
                           CIN: tl.constexpr, COUT: tl.constexpr, BLOCK_CO: tl.constexpr, BLOCK_CI: tl.constexpr,
                           BLOCK_P: tl.constexpr, ROWS_PER_PROG: tl.constexpr):
    """3x3 stride-1 pad-1 weight gradient straight from unpadded NHWC tensors. One program = one kernel row kh, all 3 kw
    taps (gy tile reused 3x). Border handling is a per-pixel mask (computed once per pixel row, not per element)."""
    pid = tl.program_id(0)
    kh = tl.program_id(1)
    cb = tl.program_id(2)
    n_ci_blocks: tl.constexpr = CIN // BLOCK_CI
    offs_co = (cb // n_ci_blocks) * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci = (cb % n_ci_blocks) * BLOCK_CI + tl.arange(0, BLOCK_CI)
    a0 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    a1 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    a2 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    start = pid * ROWS_PER_PROG
    for r in range(0, ROWS_PER_PROG, BLOCK_P):
        p = start + r + tl.arange(0, BLOCK_P)
        ok = p < P
        w = p % W
        hh = (p // W) % H + kh - 1
        okh = ok & (hh >= 0) & (hh < H)
        g = tl.trans(tl.load(gy_ptr + p[:, None] * COUT + offs_co[None, :], mask=ok[:, None], other=0.0))
        xrow = p + (kh - 1) * W            # same n, row hh, column w  (flattened NHWC pixel index)
        m0 = okh & (w >= 1)
        m2 = okh & (w < W - 1)
        x0 = tl.load(x_ptr + (xrow - 1)[:, None] * CIN + offs_ci[None, :], mask=m0[:, None], other=0.0)
        x1 = tl.load(x_ptr + xrow[:, None] * CIN + offs_ci[None, :], mask=okh[:, None], other=0.0)
        x2 = tl.load(x_ptr + (xrow + 1)[:, None] * CIN + offs_ci[None, :], mask=m2[:, None], other=0.0)
        a0 += tl.dot(g, x0)
        a1 += tl.dot(g, x1)
        a2 += tl.dot(g, x2)
    out = dw_ptr + (offs_co[:, None] * 9 + kh * 3) * CIN + offs_ci[None, :]
    tl.atomic_add(out, a0, sem='relaxed')
    tl.atomic_add(out + CIN, a1, sem='relaxed')
    tl.atomic_add(out + 2 * CIN, a2, sem='relaxed')


def wgrad3x3(x, gy, weight_shape, padding=1, *, rows_per_prog=8192, block_p=32, block_co=64, block_ci=64,
             num_warps=8, num_stages=2):
    """3x3 / stride 1 / pad 1 weight gradient, no padded copies. Returns fp32 dW with channels_last strides."""
    cout, cin, ks, _ = weight_shape
    assert ks == 3 and padding == 1 and cin % block_ci == 0 and cout % block_co == 0
    x = x.contiguous(memory_format=torch.channels_last)
    gy = gy.contiguous(memory_format=torch.channels_last)
    N, _, H, W = x.shape
    P = N * H * W
    dw = torch.zeros((cout, 9, cin), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(P, rows_per_prog), 3, (cout // block_co) * (cin // block_ci))
    _wgrad3x3_nopad_kernel[grid](x, gy, dw, P, H, W, CIN=cin, COUT=cout, BLOCK_CO=block_co, BLOCK_CI=block_ci,
                                 BLOCK_P=block_p, ROWS_PER_PROG=rows_per_prog, num_warps=num_warps,
                                 num_stages=num_stages)
    return dw.view(cout, 3, 3, cin).permute(0, 3, 1, 2)


@triton.jit
def _wgrad3x3_s2_kernel(x_ptr, gy_ptr, dw_ptr, P, H, W, Ho, Wo,
                        CIN: tl.constexpr, COUT: tl.constexpr, BLOCK_CO: tl.constexpr, BLOCK_CI: tl.constexpr,
                        BLOCK_P: tl.constexpr, ROWS_PER_PROG: tl.constexpr):
    """3x3 / stride 2 / pad 1 weight gradient from unpadded NHWC tensors (ResNet layerN.0.conv1). Same layout as
    _wgrad3x3_nopad_kernel: one program = one kernel row kh, all 3 kw taps; per-pixel border masks."""
    pid = tl.program_id(0)
    kh = tl.program_id(1)
    cb = tl.program_id(2)
    n_ci_blocks: tl.constexpr = CIN // BLOCK_CI
    offs_co = (cb // n_ci_blocks) * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci = (cb % n_ci_blocks) * BLOCK_CI + tl.arange(0, BLOCK_CI)
    a0 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    a1 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    a2 = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    start = pid * ROWS_PER_PROG
    for r in range(0, ROWS_PER_PROG, BLOCK_P):
        p = start + r + tl.arange(0, BLOCK_P)
        ok = p < P
        wo = p % Wo
        t = p // Wo
        ho = t % Ho
        n = t // Ho
        hh = ho * 2 + kh - 1
        okh = ok & (hh >= 0) & (hh < H)
        w0 = wo * 2 - 1
        base = (n * H + hh) * W + w0                     # x pixel for kw = 0
        g = tl.trans(tl.load(gy_ptr + p[:, None] * COUT + offs_co[None, :], mask=ok[:, None], other=0.0))
        m0 = okh & (w0 >= 0)
        m2 = okh & (w0 + 2 < W)
        x0 = tl.load(x_ptr + base[:, None] * CIN + offs_ci[None, :], mask=m0[:, None], other=0.0)
        x1 = tl.load(x_ptr + (base + 1)[:, None] * CIN + offs_ci[None, :], mask=okh[:, None], other=0.0)
        x2 = tl.load(x_ptr + (base + 2)[:, None] * CIN + offs_ci[None, :], mask=m2[:, None], other=0.0)
        a0 += tl.dot(g, x0)
        a1 += tl.dot(g, x1)
        a2 += tl.dot(g, x2)
    out = dw_ptr + (offs_co[:, None] * 9 + kh * 3) * CIN + offs_ci[None, :]
    tl.atomic_add(out, a0, sem='relaxed')
    tl.atomic_add(out + CIN, a1, sem='relaxed')
    tl.atomic_add(out + 2 * CIN, a2, sem='relaxed')


def wgrad3x3_s2(x, gy, weight_shape, padding=1, *, rows_per_prog=4096, block_p=32, block_co=64, block_ci=64,
                num_warps=4, num_stages=2):
    """3x3 / stride 2 / pad 1 weight gradient. Returns fp32 dW with channels_last strides."""
    cout, cin, ks, _ = weight_shape
    assert ks == 3 and padding == 1 and cin % block_ci == 0 and cout % block_co == 0
    x = x.contiguous(memory_format=torch.channels_last)
    gy = gy.contiguous(memory_format=torch.channels_last)
    N, _, H, W = x.shape
    Ho, Wo = gy.shape[2], gy.shape[3]
    P = N * Ho * Wo
    dw = torch.zeros((cout, 9, cin), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(P, rows_per_prog), 3, (cout // block_co) * (cin // block_ci))
    _wgrad3x3_s2_kernel[grid](x, gy, dw, P, H, W, Ho, Wo, CIN=cin, COUT=cout, BLOCK_CO=block_co, BLOCK_CI=block_ci,
                              BLOCK_P=block_p, ROWS_PER_PROG=rows_per_prog, num_warps=num_warps,
                              num_stages=num_stages)
    return dw.view(cout, 3, 3, cin).permute(0, 3, 1, 2)
