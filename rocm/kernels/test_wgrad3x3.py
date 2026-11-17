"""Pad-free 3x3 wgrad (conv_wgrad_s1.wgrad3x3) vs MIOpen per ResNet18 stride-1 3x3 layer, interleaved, plus best config."""
import sys, time, itertools, torch
from conv_wgrad_s1 import wgrad3x3
torch.manual_seed(0); B = 128; cl = torch.channels_last
SH = [('layer1', 64, 45, 80), ('layer2', 128, 23, 40), ('layer3', 256, 12, 20), ('layer4', 512, 6, 10)]
def t(fn, n=20):
    fn(); torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - s) / n * 1e3
for name, c, H, W in SH:
    x = torch.randn(B, c, H, W, device='cuda', dtype=torch.bfloat16).contiguous(memory_format=cl)
    w = torch.randn(c, c, 3, 3, device='cuda', dtype=torch.bfloat16).contiguous(memory_format=cl)
    gy = torch.randn(B, c, H, W, device='cuda', dtype=torch.bfloat16).contiguous(memory_format=cl)
    mi = lambda: torch.ops.aten.convolution_backward(gy, x, w, None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1, (False, True, False))[1]
    truth = torch.ops.aten.convolution_backward(gy.float(), x.float(), w.float(), None, (1, 1), (1, 1), (1, 1), False, (0, 0), 1, (False, True, False))[1]
    rel = lambda a: ((a.float() - truth).norm() / truth.norm()).item()
    fl = 2 * B * H * W * c * c * 9
    res = []
    for rpp, bp, bco, bci, nw, ns in itertools.product((1024, 2048, 4096, 8192, 16384), (32, 64), (64, 128), (64, 128), (4, 8), (1, 2)):
        if bco > c or bci > c: continue
        f = lambda: wgrad3x3(x, gy, w.shape, 1, rows_per_prog=rpp, block_p=bp, block_co=bco, block_ci=bci, num_warps=nw, num_stages=ns)
        try: o = f()
        except Exception: continue
        a = t(mi); b = t(f); res.append((b / a, b, a, rel(o), (rpp, bp, bco, bci, nw, ns)))
    res.sort(); r = res[0]
    print(f'{name} {c}x{c} {H}x{W}: miopen {r[2]:.2f} ms ({fl/r[2]/1e9:.1f} TF, err {rel(mi()):.0e}) | nopad {r[1]:.2f} ms ({fl/r[1]/1e9:.1f} TF, err {r[3]:.0e}) '
          f'speedup {1/r[0]:.2f}x cfg rpp,bp,bco,bci,warps,stages={r[4]}', flush=True)
