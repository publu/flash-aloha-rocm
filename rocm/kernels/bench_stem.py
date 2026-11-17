"""Stem conv (ResNet18 conv1: 3->64, 7x7, stride 2, pad 3, no bias) fwd + weight-grad: MIOpen vs alternatives, bf16."""
import torch, time, torch.nn.functional as F
torch.manual_seed(0)
B, H, W = 128, 180, 320
dev = 'cuda'; dt = torch.bfloat16
x = torch.randn(B, 3, H, W, device=dev, dtype=dt).contiguous(memory_format=torch.channels_last)
w = (torch.randn(64, 3, 7, 7, device=dev, dtype=dt) * .05).contiguous(memory_format=torch.channels_last)
Ho, Wo = 90, 160
gy = torch.randn(B, 64, Ho, Wo, device=dev, dtype=dt).contiguous(memory_format=torch.channels_last)

def t(fn, n=10):
    fn(); torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - s) / n * 1e3

ref_y = F.conv2d(x, w, stride=2, padding=3)
ref_gw = torch.ops.aten.convolution_backward(gy, x, w, None, (2, 2), (3, 3), (1, 1), False, (0, 0), 1, (False, True, False))[1]
print('miopen NHWC fwd %.2f ms  wgrad %.2f ms' % (t(lambda: F.conv2d(x, w, stride=2, padding=3)),
      t(lambda: torch.ops.aten.convolution_backward(gy, x, w, None, (2, 2), (3, 3), (1, 1), False, (0, 0), 1, (False, True, False)))))
xc, wc, gyc = x.contiguous(), w.contiguous(), gy.contiguous()
print('miopen NCHW fwd %.2f ms  wgrad %.2f ms' % (t(lambda: F.conv2d(xc, wc, stride=2, padding=3)),
      t(lambda: torch.ops.aten.convolution_backward(gyc, xc, wc, None, (2, 2), (3, 3), (1, 1), False, (0, 0), 1, (False, True, False)))))

# unfold + GEMM (same math; fp32 accumulation inside the GEMM)
def unfold(x):  # -> (B*Ho*Wo, 147) in (c, kh, kw) order matching w.reshape(64,147)
    return F.unfold(x, 7, padding=3, stride=2).transpose(1, 2).reshape(-1, 147)
def fwd_gemm(x, w):
    return (unfold(x) @ w.reshape(64, 147).t()).view(B, Ho, Wo, 64).permute(0, 3, 1, 2)
def wgrad_gemm(x, gy):
    g2 = gy.permute(0, 2, 3, 1).reshape(-1, 64)        # NHWC view, no copy for channels_last
    return (g2.t() @ unfold(x)).view(64, 3, 7, 7)
y2 = fwd_gemm(x, w); gw2 = wgrad_gemm(x, gy)
rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()
print('gemm fwd %.2f ms (rel err %.2e)  wgrad %.2f ms (rel err %.2e)' % (t(lambda: fwd_gemm(x, w)), rel(y2, ref_y),
      t(lambda: wgrad_gemm(x, gy)), rel(gw2, ref_gw)))
# fp32 reference for the weight grad, to compare both bf16 paths against ground truth
gw32 = torch.ops.aten.convolution_backward(gy.float().contiguous(), x.float().contiguous(), w.float().contiguous(), None, (2, 2), (3, 3), (1, 1), False, (0, 0), 1, (False, True, False))[1]
y32 = F.conv2d(x.float().contiguous(), w.float().contiguous(), stride=2, padding=3)
print('vs fp32 truth: miopen fwd %.2e gemm fwd %.2e | miopen wgrad %.2e gemm wgrad %.2e' % (rel(ref_y, y32), rel(y2, y32), rel(ref_gw, gw32), rel(gw2, gw32)))
