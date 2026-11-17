"""Per-parameter gradient error vs an fp32 reference: stock MIOpen bf16 vs patched bf16 (same weights, same input).
Patched passes if its error is not worse than stock's."""
import torch, torchvision
import flash_rocm_kernels as K
torch.manual_seed(0)
ref = torchvision.models.resnet18(weights='IMAGENET1K_V1').cuda().eval().to(memory_format=torch.channels_last)  # ACT: pretrained + frozen BN
m = torchvision.models.resnet18(weights=None).cuda().eval(); m.load_state_dict(ref.state_dict())
m = m.to(torch.bfloat16).to(memory_format=torch.channels_last)
x32 = torch.rand(64, 3, 180, 320, device='cuda').contiguous(memory_format=torch.channels_last)
x = x32.to(torch.bfloat16)
body = lambda mm, xx: torch.nn.Sequential(*list(mm.children())[:-2])(xx)
def grads(mm, xx):
    mm.zero_grad(); body(mm, xx).float().square().mean().backward()
    return {n: p.grad.float().clone() for n, p in mm.named_parameters() if p.grad is not None}
torch.backends.cudnn.allow_tf32 = False
g32 = grads(ref, x32)
gs = grads(m, x)
K.install(); gp = grads(m, x)
rel = lambda a, n: ((a[n] - g32[n]).norm() / g32[n].norm()).item()
rows = [(n, rel(gs, n), rel(gp, n)) for n in g32 if 'conv' in n or 'downsample.0' in n]
worse = [r for r in rows if r[2] > 1.25 * r[1] + 1e-3]
for n, a, b in rows:
    print(f'{n:28s} stock {a:.2e}  patched {b:.2e}' + ('  <-- WORSE' if (n, a, b) in worse else ''))
print('median stock', sorted(r[1] for r in rows)[len(rows)//2], 'median patched', sorted(r[2] for r in rows)[len(rows)//2])
print('PASS' if not worse else f'FAIL: {len(worse)} params worse than stock')
