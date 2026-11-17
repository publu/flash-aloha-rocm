"""Flipped, transposed conv weight for the input gradient: out = w.flip(2,3).transpose(0,1), channels_last.
channels_last memory of w is (co, tap, ci); target memory is (ci, T-1-tap, co): a tiled 2D transpose per tap."""
import torch
import triton
import triton.language as tl


@triton.jit
def _flipT_kernel(w_ptr, o_ptr, CO: tl.constexpr, CI: tl.constexpr, T: tl.constexpr,
                  BCO: tl.constexpr, BCI: tl.constexpr):
    tap = tl.program_id(0)
    co = tl.program_id(1) * BCO + tl.arange(0, BCO)
    ci = tl.program_id(2) * BCI + tl.arange(0, BCI)
    m = (co[:, None] < CO) & (ci[None, :] < CI)
    tile = tl.load(w_ptr + (co[:, None] * T + tap) * CI + ci[None, :], mask=m)
    tl.store(o_ptr + (ci[None, :] * T + (T - 1 - tap)) * CO + co[:, None], tile, mask=m)


def flip_transpose(w):
    co, ci, kh, kw = w.shape
    w = w.contiguous(memory_format=torch.channels_last)
    out = torch.empty((ci, co, kh, kw), device=w.device, dtype=w.dtype, memory_format=torch.channels_last)
    B = 64
    _flipT_kernel[(kh * kw, triton.cdiv(co, B), triton.cdiv(ci, B))](w, out, CO=co, CI=ci, T=kh * kw, BCO=B, BCI=B,
                                                                     num_warps=4)
    return out
