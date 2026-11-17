import sys, time, torch
sys.path.insert(0, sys.argv[1])
from stem_wgrad import stem_wgrad
torch.manual_seed(0)
B, H, W = 128, 180, 320
x = torch.randn(B, 3, H, W, device='cuda', dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
w = torch.randn(64, 3, 7, 7, device='cuda', dtype=torch.bfloat16)
gy = torch.randn(B, 64, 90, 160, device='cuda', dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
cb = lambda xx, gg, ww: torch.ops.aten.convolution_backward(gg, xx, ww, None, (2, 2), (3, 3), (1, 1), False, (0, 0), 1, (False, True, False))[1]
truth = cb(x.float().contiguous(), gy.float().contiguous(), w.float())
mi = cb(x, gy, w.contiguous(memory_format=torch.channels_last))
rel = lambda a: ((a.float() - truth).norm() / truth.norm()).item()
def t(fn, n=20):
    fn(); torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - s) / n * 1e3
print('miopen  %.2f ms  err %.2e' % (t(lambda: cb(x, gy, w.contiguous(memory_format=torch.channels_last)), 5), rel(mi)))
import itertools
res=[]
for rpp, bp, bk, nw, ns in itertools.product((2048, 4096, 8192), (32, 64, 128), (32, 64), (4, 8), (1, 2)):
    f = lambda: stem_wgrad(x, gy, w.shape, 2, 3, rows_per_prog=rpp, block_p=bp, block_k=bk, num_warps=nw, num_stages=ns)
    try:
        out = f(); ms = t(f, 10); res.append((ms, rpp, bp, bk, nw, ns, rel(out)))
    except Exception as e:
        pass
res.sort()
for r in res[:8]: print('triton %.2f ms  rpp=%d bp=%d bk=%d warps=%d stages=%d err %.2e' % r)
print('configs tried', len(res))
