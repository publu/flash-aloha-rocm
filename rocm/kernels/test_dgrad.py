"""Input-gradient (dgrad) per ResNet18 layer: MIOpen/CK backward-data vs mathematically identical alternatives.
stride 1: dx = conv2d(gy, flip(w).transpose(0,1), pad=ks-1-p)   (forward-conv kernels)
1x1 stride s: dx[:, :, ::s, ::s] = gy @ w ; zeros elsewhere         (one GEMM)
Interleaved timing (GPU is shared)."""
import time, torch, torch.nn.functional as F
torch.manual_seed(0); B = 128
SH = [('layer1 64->64 3x3', 64, 64, 3, 1, 1, 45, 80), ('layer2 128->128 3x3', 128, 128, 3, 1, 1, 23, 40),
      ('layer3 256->256 3x3', 256, 256, 3, 1, 1, 12, 20), ('layer4 512->512 3x3', 512, 512, 3, 1, 1, 6, 10),
      ('layer2.0 64->128 3x3/2', 64, 128, 3, 2, 1, 45, 80), ('layer3.0 128->256 3x3/2', 128, 256, 3, 2, 1, 23, 40),
      ('layer4.0 256->512 3x3/2', 256, 512, 3, 2, 1, 12, 20),
      ('layer2 down 1x1/2', 64, 128, 1, 2, 0, 45, 80), ('layer3 down 1x1/2', 128, 256, 1, 2, 0, 23, 40),
      ('layer4 down 1x1/2', 256, 512, 1, 2, 0, 12, 20)]
def t(fn, n=10):
    fn(); torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - s) / n * 1e3
cl = torch.channels_last
for name, cin, cout, ks, st, pd, H, W in SH:
    x = torch.randn(B, cin, H, W, device='cuda', dtype=torch.bfloat16).contiguous(memory_format=cl)
    w = (torch.randn(cout, cin, ks, ks, device='cuda', dtype=torch.bfloat16) * 0.05).contiguous(memory_format=cl)
    Ho = (H + 2 * pd - ks) // st + 1; Wo = (W + 2 * pd - ks) // st + 1
    gy = torch.randn(B, cout, Ho, Wo, device='cuda', dtype=torch.bfloat16).contiguous(memory_format=cl)
    mi = lambda: torch.ops.aten.convolution_backward(gy, x, w, None, (st, st), (pd, pd), (1, 1), False, (0, 0), 1, (True, False, False))[0]
    truth = torch.ops.aten.convolution_backward(gy.float(), x.float(), w.float(), None, (st, st), (pd, pd), (1, 1), False, (0, 0), 1, (True, False, False))[0]
    rel = lambda a: ((a.float() - truth).norm() / truth.norm()).item()
    alts = {}
    if st == 1:
        wf = w.flip(2, 3).transpose(0, 1).contiguous(memory_format=cl)
        alts['fwd-conv'] = lambda: F.conv2d(gy, wf, None, 1, ks - 1 - pd)
        alts['fwd-conv+flip'] = lambda: F.conv2d(gy, w.flip(2, 3).transpose(0, 1).contiguous(memory_format=cl), None, 1, ks - 1 - pd)
    if ks == 1:
        w2 = w.view(cout, cin)
        def pw():
            dx = torch.zeros(B, H, W, cin, device='cuda', dtype=torch.bfloat16)
            dx[:, :Ho * st:st, :Wo * st:st, :] = (gy.permute(0, 2, 3, 1).reshape(-1, cout) @ w2).view(B, Ho, Wo, cin)
            return dx.permute(0, 3, 1, 2)
        alts['gemm-scatter'] = pw
    if st == 2 and ks == 3:
        alts['conv_transpose'] = lambda: F.conv_transpose2d(gy, w, None, 2, 1, output_padding=(H - ((Ho - 1) * 2 - 2 + 3), W - ((Wo - 1) * 2 - 2 + 3)))
    out = [f'{name:26s} miopen {t(mi):6.2f} ms err {rel(mi()):.0e}']
    for k, f in alts.items():
        try:
            a = t(mi); b = t(f); out.append(f'{k} {b:6.2f} ms ({a/b:.2f}x vs {a:.2f}) err {rel(f()):.0e}')
        except Exception as e:
            out.append(f'{k} FAIL {str(e)[:60]}')
    print(' | '.join(out), flush=True)
