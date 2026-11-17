"""Triton fwd 3x3 vs MIOpen/CK forward per ResNet18 stride-1 3x3 layer (interleaved), best config per layer."""
import itertools, time, torch, torch.nn.functional as F
from conv3x3_fwd import conv3x3_fwd
torch.manual_seed(0); B = 128; cl = torch.channels_last
SH = [('layer1', 64, 45, 80), ('layer2', 128, 23, 40), ('layer3', 256, 12, 20), ('layer4', 512, 6, 10)]
def t(fn, n=20):
    fn(); torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - s) / n * 1e3
for name, c, H, W in SH:
    x = torch.randn(B, c, H, W, device='cuda', dtype=torch.bfloat16).contiguous(memory_format=cl)
    w = (torch.randn(c, c, 3, 3, device='cuda', dtype=torch.bfloat16) * 0.05).contiguous(memory_format=cl)
    mi = lambda: F.conv2d(x, w, None, 1, 1)
    truth = F.conv2d(x.float(), w.float(), None, 1, 1)
    rel = lambda a: ((a.float() - truth).norm() / truth.norm()).item()
    fl = 2 * B * H * W * c * c * 9
    res = []
    for bp, bco, bci, g, nw, ns in itertools.product((64, 128, 256), (64, 128), (32, 64), (1, 8), (4, 8), (1, 2)):
        if bco > c or bci > c: continue
        f = lambda: conv3x3_fwd(x, w, block_p=bp, block_co=bco, block_ci=bci, group_p=g, num_warps=nw, num_stages=ns)
        try: o = f()
        except Exception: continue
        a = t(mi); b = t(f); res.append((b / a, b, a, rel(o), (bp, bco, bci, g, nw, ns)))
    res.sort(); r = res[0]
    print(f'{name} {c}: miopen {r[2]:.2f} ms ({fl/r[2]/1e9:.1f} TF, err {rel(mi()):.0e}) | triton {r[1]:.2f} ms ({fl/r[1]/1e9:.1f} TF, err {r[3]:.0e}) '
          f'speedup {1/r[0]:.2f}x cfg bp,bco,bci,group,warps,stages={r[4]}', flush=True)
