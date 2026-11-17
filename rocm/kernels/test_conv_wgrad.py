import sys, time, itertools, torch
from conv_wgrad import conv_wgrad
torch.manual_seed(0)
B = 128
# (name, cin, cout, ks, stride, pad, H, W) -- ResNet18 layers at 180x320 input
SHAPES = [('stem 3->64 7x7/2', 3, 64, 7, 2, 3, 180, 320), ('layer1 64->64 3x3', 64, 64, 3, 1, 1, 45, 80),
          ('layer2.0 64->128 3x3/2', 64, 128, 3, 2, 1, 45, 80), ('layer2 down 64->128 1x1/2', 64, 128, 1, 2, 0, 45, 80),
          ('layer2 128->128 3x3', 128, 128, 3, 1, 1, 23, 40),
          ('layer3.0 128->256 3x3/2', 128, 256, 3, 2, 1, 23, 40), ('layer3 down 128->256 1x1/2', 128, 256, 1, 2, 0, 23, 40),
          ('layer3 256->256 3x3', 256, 256, 3, 1, 1, 12, 20), ('layer4.0 256->512 3x3/2', 256, 512, 3, 2, 1, 12, 20),
          ('layer4 down 256->512 1x1/2', 256, 512, 1, 2, 0, 12, 20), ('layer4 512->512 3x3', 512, 512, 3, 1, 1, 6, 10)]
only = sys.argv[1] if len(sys.argv) > 1 else None
def t(fn, n=10):
    fn(); torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - s) / n * 1e3
for name, cin, cout, ks, st, pd, H, W in SHAPES:
    if only and not any(o in name for o in only.split(',')): continue
    x = torch.randn(B, cin, H, W, device='cuda', dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
    w = torch.randn(cout, cin, ks, ks, device='cuda', dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
    Ho = (H + 2 * pd - ks) // st + 1; Wo = (W + 2 * pd - ks) // st + 1
    gy = torch.randn(B, cout, Ho, Wo, device='cuda', dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
    cb = lambda xx, gg, ww: torch.ops.aten.convolution_backward(gg, xx, ww, None, (st, st), (pd, pd), (1, 1), False, (0, 0), 1, (False, True, False))[1]
    truth = cb(x.float().contiguous(), gy.float().contiguous(), w.float().contiguous())
    rel = lambda a: ((a.float() - truth).norm() / truth.norm()).item()
    flops = 2 * B * Ho * Wo * cout * cin * ks * ks
    mi_ms = t(lambda: cb(x, gy, w)); mi_err = rel(cb(x, gy, w))
    res = []
    for rpp, bp, bco, bk, nw in itertools.product((256, 1024, 2048, 4096), (32, 64), (32, 64, 128), (32, 64, 128), (4, 8)):
        if bco > cout: continue
        f = lambda: conv_wgrad(x, gy, w.shape, st, pd, rows_per_prog=rpp, block_p=bp, block_co=bco, block_k=bk, num_warps=nw)
        try:
            out = f(); res.append((t(f), rpp, bp, bco, bk, nw, rel(out)))
        except Exception:
            pass
    res.sort()
    b = res[0]
    print(f'{name:28s} miopen {mi_ms:6.2f} ms ({flops/mi_ms/1e9:5.1f} TF, err {mi_err:.1e}) | triton {b[0]:6.2f} ms ({flops/b[0]/1e9:5.1f} TF, err {b[6]:.1e}) '
          f'rpp={b[1]} bp={b[2]} bco={b[3]} bk={b[4]} warps={b[5]}  speedup {mi_ms/b[0]:.2f}x', flush=True)
