import sys, time, itertools, torch
from conv_wgrad_s1 import shift_wgrad, shift_wgrad_bmm, pointwise_wgrad
torch.manual_seed(0); B = 128
S1 = [('layer1 64->64 3x3', 64, 64, 45, 80), ('layer2 128->128 3x3', 128, 128, 23, 40),
      ('layer3 256->256 3x3', 256, 256, 12, 20), ('layer4 512->512 3x3', 512, 512, 6, 10)]
PW = [('layer2 down 64->128 1x1/2', 64, 128, 45, 80), ('layer3 down 128->256 1x1/2', 128, 256, 23, 40),
      ('layer4 down 256->512 1x1/2', 256, 512, 12, 20)]
def t(fn, n=10):
    fn(); torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - s) / n * 1e3
def setup(cin, cout, ks, st, pd, H, W):
    x = torch.randn(B, cin, H, W, device='cuda', dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
    w = torch.randn(cout, cin, ks, ks, device='cuda', dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
    Ho = (H + 2 * pd - ks) // st + 1; Wo = (W + 2 * pd - ks) // st + 1
    gy = torch.randn(B, cout, Ho, Wo, device='cuda', dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
    cb = lambda xx, gg, ww: torch.ops.aten.convolution_backward(gg, xx, ww, None, (st, st), (pd, pd), (1, 1), False, (0, 0), 1, (False, True, False))[1]
    truth = cb(x.float().contiguous(), gy.float().contiguous(), w.float().contiguous())
    rel = lambda a: ((a.float() - truth).norm() / truth.norm()).item()
    return x, w, gy, cb, rel, 2 * B * Ho * Wo * cout * cin * ks * ks
for name, cin, cout, H, W in S1:
    x, w, gy, cb, rel, fl = setup(cin, cout, 3, 1, 1, H, W)
    mi = t(lambda: cb(x, gy, w)); bm = t(lambda: shift_wgrad_bmm(x, gy, w.shape, 1))
    res = []
    for rpp, bp, bco, bci, nw, ns in itertools.product((512, 1024, 2048, 4096, 8192), (32, 64), (64, 128), (64, 128), (4, 8), (1, 2)):
        if bco > cout or bci > cin: continue
        f = lambda: shift_wgrad(x, gy, w.shape, 1, rows_per_prog=rpp, block_p=bp, block_co=bco, block_ci=bci, num_warps=nw, num_stages=ns)
        try: o = f(); res.append((t(f), rpp, bp, bco, bci, nw, ns, rel(o)))
        except Exception as e: pass
    b = min(res)
    print(f'{name:28s} miopen {mi:6.2f} ({fl/mi/1e9:4.1f}TF err {rel(cb(x,gy,w)):.0e}) | per-tap mm {bm:6.2f} ({fl/bm/1e9:4.1f}TF err {rel(shift_wgrad_bmm(x,gy,w.shape,1)):.0e}) '
          f'| triton-shift {b[0]:6.2f} ({fl/b[0]/1e9:4.1f}TF err {b[7]:.0e}) cfg={b[1:7]}', flush=True)
for name, cin, cout, H, W in PW:
    x, w, gy, cb, rel, fl = setup(cin, cout, 1, 2, 0, H, W)
    mi = t(lambda: cb(x, gy, w)); pw = t(lambda: pointwise_wgrad(x, gy, w.shape, 2))
    print(f'{name:28s} miopen {mi:6.2f} ({fl/mi/1e9:4.1f}TF) | gemm {pw:6.2f} ({fl/pw/1e9:4.1f}TF err {rel(pointwise_wgrad(x,gy,w.shape,2)):.0e})', flush=True)
