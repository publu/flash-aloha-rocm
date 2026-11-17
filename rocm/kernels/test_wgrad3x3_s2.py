"""Stride-2 3x3 wgrad (wgrad3x3_s2) vs MIOpen, ResNet18 layerN.0.conv1 shapes; plus wider retune of stride-1 wgrad3x3."""
import itertools, sys, time, torch
from conv_wgrad_s1 import wgrad3x3_s2, wgrad3x3
torch.manual_seed(0); B = 128; cl = torch.channels_last
def t(fn, n=20):
    fn(); torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - s) / n * 1e3
def run(name, cin, cout, H, W, st, fn, space):
    x = torch.randn(B, cin, H, W, device='cuda', dtype=torch.bfloat16).contiguous(memory_format=cl)
    w = torch.randn(cout, cin, 3, 3, device='cuda', dtype=torch.bfloat16).contiguous(memory_format=cl)
    Ho, Wo = (H + 2 - 3) // st + 1, (W + 2 - 3) // st + 1
    gy = torch.randn(B, cout, Ho, Wo, device='cuda', dtype=torch.bfloat16).contiguous(memory_format=cl)
    mi = lambda: torch.ops.aten.convolution_backward(gy, x, w, None, (st, st), (1, 1), (1, 1), False, (0, 0), 1, (False, True, False))[1]
    truth = torch.ops.aten.convolution_backward(gy.float(), x.float(), w.float(), None, (st, st), (1, 1), (1, 1), False, (0, 0), 1, (False, True, False))[1]
    rel = lambda a: ((a.float() - truth).norm() / truth.norm()).item()
    fl = 2 * B * Ho * Wo * cout * cin * 9
    res = []
    for rpp, bp, bco, bci, nw, ns in space:
        if bco > cout or bci > cin or bp > rpp: continue
        f = lambda: fn(x, gy, w.shape, 1, rows_per_prog=rpp, block_p=bp, block_co=bco, block_ci=bci, num_warps=nw, num_stages=ns)
        try: o = f()
        except Exception: continue
        a = t(mi); b = t(f); res.append((b / a, b, a, rel(o), (rpp, bp, bco, bci, nw, ns)))
    res.sort(); r = res[0]
    print(f'{name}: miopen {r[2]:.2f} ms ({fl/r[2]/1e9:.1f} TF, err {rel(mi()):.0e}) | triton {r[1]:.2f} ms ({fl/r[1]/1e9:.1f} TF, err {r[3]:.0e}) '
          f'speedup {1/r[0]:.2f}x cfg rpp,bp,bco,bci,warps,stages={r[4]}', flush=True)
S = list(itertools.product((1024, 2048, 4096, 8192, 16384), (32, 64, 128), (64, 128), (64, 128), (4, 8), (1, 2, 3)))
which = sys.argv[1] if len(sys.argv) > 1 else 's2,s1'
if 's2' in which:
    run('layer2.0 64->128 /2', 64, 128, 45, 80, 2, wgrad3x3_s2, S)
    run('layer3.0 128->256 /2', 128, 256, 23, 40, 2, wgrad3x3_s2, S)
    run('layer4.0 256->512 /2', 256, 512, 12, 20, 2, wgrad3x3_s2, S)
if 's1' in which:
    for n, c, H, W in [('layer1', 64, 45, 80), ('layer2', 128, 23, 40), ('layer3', 256, 12, 20), ('layer4', 512, 6, 10)]:
        run(n + ' s1 retune', c, c, H, W, 1, wgrad3x3, S)
