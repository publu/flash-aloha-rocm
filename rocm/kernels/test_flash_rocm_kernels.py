"""Real ResNet18 (ImageNet weights as Flash loads them) fwd+bwd: patched vs MIOpen grads, and timing, interleaved."""
import os, sys, time, copy, torch, torchvision
import flash_rocm_kernels as K
torch.manual_seed(0)
m0 = torchvision.models.resnet18(weights=None).cuda().to(torch.bfloat16).to(memory_format=torch.channels_last)
x = torch.randn(128, 3, 180, 320, device='cuda', dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
feat = lambda m: torch.nn.Sequential(*list(m.children())[:-2])(x)
def grads(m):
    m.zero_grad(); feat(m).float().square().mean().backward()
    return {n: p.grad.float().clone() for n, p in m.named_parameters() if p.grad is not None}
g0 = grads(m0)
orig = torch.nn.Conv2d.forward
K.install()
g1 = grads(m0)
worst = max(((g1[n] - g0[n]).norm() / g0[n].norm()).item() for n in g0)
print('max rel grad diff patched vs MIOpen (bf16):', f'{worst:.2e}', 'conv1:', f"{((g1['conv1.weight']-g0['conv1.weight']).norm()/g0['conv1.weight'].norm()).item():.2e}",
      'layer1:', f"{((g1['layer1.0.conv1.weight']-g0['layer1.0.conv1.weight']).norm()/g0['layer1.0.conv1.weight'].norm()).item():.2e}")
patched = torch.nn.Conv2d.forward
def step(): m0.zero_grad(); feat(m0).float().square().mean().backward()
def t(n=5):
    step(); torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(n): step()
    torch.cuda.synchronize(); return (time.perf_counter() - s) / n * 1e3
for r in range(3):
    torch.nn.Conv2d.forward = orig; a = t()
    torch.nn.Conv2d.forward = patched; b = t()
    print(f'round {r}: MIOpen {a:.1f} ms  patched {b:.1f} ms  speedup {a/b:.2f}x (one camera, fwd+bwd, batch 128)', flush=True)
