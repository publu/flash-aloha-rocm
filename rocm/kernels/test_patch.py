import torch, torchvision, time, copy
import flash_rocm_kernels as K
torch.manual_seed(0)
m = torchvision.models.resnet18(weights=None).cuda().to(torch.bfloat16).to(memory_format=torch.channels_last)
ref = copy.deepcopy(m)
x = torch.randn(128, 3, 180, 320, device='cuda', dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
def grads(model, compiled):
    f = torch.compile(model) if compiled else model
    model.zero_grad(); y = f(x).float().pow(2).mean(); y.backward()
    return {n: p.grad.float().clone() for n, p in model.named_parameters()}
def tm(model, compiled, n=5):
    f = torch.compile(model) if compiled else model
    for _ in range(2): model.zero_grad(); f(x).float().pow(2).mean().backward()
    torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(n): model.zero_grad(); f(x).float().pow(2).mean().backward()
    torch.cuda.synchronize(); return (time.perf_counter() - s) / n * 1e3
g0 = grads(ref, False); t0 = tm(ref, False)
K.install()
for comp in (False, True):
    g1 = grads(m, comp)
    worst = max(((g1[k] - g0[k]).norm() / (g0[k].norm() + 1e-12)).item() for k in g0)
    c1 = ((g1['conv1.weight'] - g0['conv1.weight']).norm() / g0['conv1.weight'].norm()).item()
    print(f'compiled={comp}: conv1.weight grad rel diff {c1:.2e}, worst param {worst:.2e}, step {tm(m, comp):.1f} ms vs original eager {t0:.1f} ms')
