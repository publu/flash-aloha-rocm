"""gfx1151 kernel substitutions for Flash ALOHA training. Same math, different kernels. Opt-in: FLASH_ROCM_KERNELS=1.

Convolution weight gradients (bf16, channels_last, groups=1, no bias) run on Triton implicit-GEMM kernels instead of
MIOpen/CK when this machine's per-layer benchmark showed Triton winning (WGRAD_CONFIGS below, from
test_conv_wgrad.py). Forward and input gradient stay on MIOpen. Exposed as torch.library custom ops so
torch.compile treats them as opaque.
  stem 3->64 7x7/2: dedicated stem kernel (MIOpen ~66 ms/camera -> ~8 ms at batch 128)
FLASH_ROCM_KERNELS=stem limits substitution to the stem only.
"""
import os
import torch
import torch.nn.functional as F

_PATCHED = False
_RAN = set()

# (cin, cout, ks, stride, pad) -> method + config, from test_shift_wgrad.py / l1 interleaved timing on gfx1151.
# Shapes not listed stay on MIOpen (layer2-4 3x3: MIOpen/CK already 8-18 TFLOPs there).
WGRAD_CONFIGS = {
    # pad-free 3x3 kernel (conv_wgrad_s1.wgrad3x3), test_wgrad3x3.py on a free GPU, speedup vs MIOpen:
    (64, 64, 3, 1, 1): ('3x3', dict(rows_per_prog=16384, block_p=32, block_co=64, block_ci=64, num_warps=4,
                                    num_stages=1)),             # layer1 4.2x
    (128, 128, 3, 1, 1): ('3x3', dict(rows_per_prog=4096, block_p=64, block_co=64, block_ci=64, num_warps=8,
                                      num_stages=2)),           # layer2 3.2x
    (256, 256, 3, 1, 1): ('3x3', dict(rows_per_prog=16384, block_p=32, block_co=64, block_ci=64, num_warps=8,
                                      num_stages=2)),           # layer3 2.3x
    (512, 512, 3, 1, 1): ('3x3', dict(rows_per_prog=1024, block_p=32, block_co=64, block_ci=128, num_warps=8,
                                      num_stages=2)),           # layer4 1.4x
    (64, 128, 1, 2, 0): ('pointwise', {}),                    # layer2 downsample: ~5x
    (128, 256, 1, 2, 0): ('pointwise', {}),                   # layer3 downsample: ~2x
}
# stride-1 'same' convs whose input gradient runs faster as a forward conv with flipped, transposed weights
# (identical math: dx = conv2d(gy, flip(w)^T, pad=ks-1-p)); test_dgrad.py on a free GPU: layer1 1.6x, layer2 1.8x,
# layer3 1.8x, layer4 1.2x -> ~2x with the Triton weight flip (wflip.py, 0.03 ms vs 0.73 ms for 512x512x3x3).
DGRAD_FWDCONV = {(64, 64, 3, 1, 1), (128, 128, 3, 1, 1), (256, 256, 3, 1, 1), (512, 512, 3, 1, 1)}


# Triton forward 3x3/s1/p1 conv (conv3x3_fwd.py), test_conv3x3_fwd.py: layer1 1.16x, layer2-4 ~2.1x vs MIOpen.
# Used for the forward pass and for the input gradient (forward conv with flip_transpose(w)); keyed by channels.
FWD3X3_CONFIGS = {
    64: dict(block_p=64, block_co=64, block_ci=64, group_p=1, num_warps=4, num_stages=1),
    128: dict(block_p=128, block_co=128, block_ci=32, group_p=1, num_warps=8, num_stages=1),
    256: dict(block_p=64, block_co=64, block_ci=64, group_p=8, num_warps=4, num_stages=2),
    512: dict(block_p=64, block_co=64, block_ci=64, group_p=1, num_warps=4, num_stages=2),
}


# Disabled: in the full compiled step CK's forward already runs ~20 TF (1.7 ms/call); Triton matched it, no gain
# (sweep6 v6-1 profile). The standalone test's 2x came from MIOpen picking a worse kernel there.
FWD3X3_ENABLED = os.environ.get('FLASH_ROCM_FWD3X3') == '1'


def _fwd3x3_cfg(w, padding, stride=1):
    if FWD3X3_ENABLED and stride == 1 and padding == 1 and tuple(w.shape[2:]) == (3, 3) and w.shape[0] == w.shape[1]:
        return FWD3X3_CONFIGS.get(int(w.shape[0]))
    return None


def _is_stem(k):
    return k[0] <= 4 and k[2:] == (7, 2, 3)


def _define_ops():
    from stem_wgrad import stem_wgrad
    from conv_wgrad_s1 import shift_wgrad, pointwise_wgrad, wgrad3x3, wgrad3x3_s2
    from wflip import flip_transpose
    from conv3x3_fwd import conv3x3_fwd

    @torch.library.custom_op('flashrocm::conv_dgrad_fwd', mutates_args=())
    def conv_dgrad_fwd(gy: torch.Tensor, w: torch.Tensor, padding: int) -> torch.Tensor:
        k = ('dgrad',) + tuple(w.shape)
        if k not in _RAN:
            _RAN.add(k)
            print(f'[flash_rocm_kernels] RAN fwd-conv dgrad {tuple(w.shape)} (pid {os.getpid()})', flush=True)
        cfg = _fwd3x3_cfg(w, padding)
        if cfg is not None:
            return conv3x3_fwd(gy, flip_transpose(w), **cfg)
        return F.conv2d(gy, flip_transpose(w), None, 1, w.shape[2] - 1 - padding)

    @conv_dgrad_fwd.register_fake
    def _(gy, w, padding):
        return gy.new_empty((gy.shape[0], w.shape[1], gy.shape[2], gy.shape[3])).contiguous(
            memory_format=torch.channels_last)

    @torch.library.custom_op('flashrocm::conv', mutates_args=())
    def conv(x: torch.Tensor, w: torch.Tensor, stride: int, padding: int) -> torch.Tensor:
        cfg = _fwd3x3_cfg(w, padding, stride)
        if cfg is not None:
            return conv3x3_fwd(x, w, **cfg)
        return F.conv2d(x, w, None, stride, padding)

    @conv.register_fake
    def _(x, w, stride, padding):
        n, _, h, ww = x.shape
        ho = (h + 2 * padding - w.shape[2]) // stride + 1
        wo = (ww + 2 * padding - w.shape[3]) // stride + 1
        return x.new_empty((n, w.shape[0], ho, wo)).contiguous(memory_format=torch.channels_last)

    @torch.library.custom_op('flashrocm::conv_wgrad', mutates_args=())
    def conv_wgrad_op(x: torch.Tensor, gy: torch.Tensor, w: torch.Tensor, stride: int, padding: int) -> torch.Tensor:
        cout, cin, ks, _ = w.shape
        k = (cin, cout, ks, stride, padding)
        if k not in _RAN:  # runtime proof: this body executes on every backward, compiled or not
            _RAN.add(k)
            print(f'[flash_rocm_kernels] RAN Triton wgrad {k} (pid {os.getpid()})', flush=True)
        if cin <= 4:
            dw = stem_wgrad(x, gy, tuple(w.shape), stride, padding, rows_per_prog=2048, block_p=32, block_k=64,
                            num_warps=4, num_stages=1)
        else:
            method, cfg = WGRAD_CONFIGS[(cin, cout, ks, stride, padding)]
            if method == '3x3':
                dw = wgrad3x3(x, gy, tuple(w.shape), padding, **cfg)
            elif method == '3x3s2':
                dw = wgrad3x3_s2(x, gy, tuple(w.shape), padding, **cfg)
            elif method == 'shift':
                dw = shift_wgrad(x, gy, tuple(w.shape), padding, **cfg)
            else:
                dw = pointwise_wgrad(x, gy, tuple(w.shape), stride)
        fmt = torch.channels_last if w.is_contiguous(memory_format=torch.channels_last) else torch.contiguous_format
        return dw.to(w.dtype).contiguous(memory_format=fmt)

    @conv_wgrad_op.register_fake
    def _(x, gy, w, stride, padding):
        return torch.empty_like(w)

    def setup_context(ctx, inputs, output):
        x, w, stride, padding = inputs
        ctx.save_for_backward(x, w)
        ctx.stride, ctx.padding = stride, padding

    def backward(ctx, gy):
        x, w = ctx.saved_tensors
        cout, cin, ks, _ = w.shape
        k = (cin, cout, ks, ctx.stride, ctx.padding)
        gx = gw = None
        if ctx.needs_input_grad[0]:
            if k in DGRAD_FWDCONV:
                gx = torch.ops.flashrocm.conv_dgrad_fwd(gy, w, ctx.padding)
            else:
                gx = torch.ops.aten.convolution_backward(gy, x, w, None, (ctx.stride,) * 2, (ctx.padding,) * 2,
                                                         (1, 1), False, (0, 0), 1, (True, False, False))[0]
        if ctx.needs_input_grad[1]:
            if _is_stem(k) or k in WGRAD_CONFIGS:
                gw = torch.ops.flashrocm.conv_wgrad(x, gy, w, ctx.stride, ctx.padding)
            else:
                gw = torch.ops.aten.convolution_backward(gy, x, w, None, (ctx.stride,) * 2, (ctx.padding,) * 2,
                                                         (1, 1), False, (0, 0), 1, (False, True, False))[1]
        return gx, gw, None, None

    conv.register_autograd(backward, setup_context=setup_context)
    return conv


def _key(m):
    ks, st, pd = m.kernel_size, m.stride, m.padding
    if not (ks[0] == ks[1] and st[0] == st[1] and isinstance(pd, tuple) and pd[0] == pd[1]):
        return None
    return (m.in_channels, m.out_channels, ks[0], st[0], pd[0])


def _eligible(m, x, stem_only):
    if not (m.dilation == (1, 1) and m.groups == 1 and m.bias is None and m.padding_mode == 'zeros'
            and x.is_cuda and x.dtype == torch.bfloat16 and m.weight.dtype == torch.bfloat16 and x.dim() == 4
            and x.is_contiguous(memory_format=torch.channels_last) and m.weight.requires_grad):
        return False
    k = _key(m)
    if k is None:
        return False
    if _is_stem(k):
        return True
    return not stem_only and (k in WGRAD_CONFIGS or k in DGRAD_FWDCONV)


def install(stem_only=False):
    global _PATCHED
    if _PATCHED:
        return
    conv = _define_ops()
    original = torch.nn.Conv2d.forward

    seen = set()

    def forward(self, x):
        if _eligible(self, x, stem_only):
            k = _key(self)
            if k not in seen and not torch.compiler.is_compiling():
                seen.add(k)
                print(f'[flash_rocm_kernels] active: conv {k} (pid {os.getpid()})', flush=True)
            return conv(x, self.weight, self.stride[0], self.padding[0])
        return original(self, x)

    torch.nn.Conv2d.forward = forward
    _PATCHED = True
    n = 0 if stem_only else len(WGRAD_CONFIGS)
    print(f'[flash_rocm_kernels] conv weight-grad -> Triton (stem + {n} tuned shapes), '
          f'dgrad -> forward conv ({0 if stem_only else len(DGRAD_FWDCONV)} shapes)', flush=True)


_mode = os.environ.get('FLASH_ROCM_KERNELS')
if _mode in ('1', 'stem'):
    install(stem_only=_mode == 'stem')
