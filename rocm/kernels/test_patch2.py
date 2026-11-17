import torch, torchvision, copy
torch.manual_seed(0)
m = torchvision.models.resnet18(weights=None).cuda().to(torch.bfloat16).to(memory_format=torch.channels_last)
x = torch.randn(128, 3, 180, 320, device='cuda', dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
def grads(model, compiled):
    model = copy.deepcopy(model)
    f = torch.compile(model) if compiled else model
    y = f(x).float().pow(2).mean(); y.backward()
    return {n: p.grad.float().clone() for n, p in model.named_parameters()}
def cmp(a, b):
    return ((a['conv1.weight'] - b['conv1.weight']).norm() / b['conv1.weight'].norm()).item(), max(((a[k] - b[k]).norm() / (b[k].norm() + 1e-12)).item() for k in b)
e = grads(m, False); c = grads(m, True)
print('UNPATCHED compiled vs eager: conv1 %.2e worst %.2e' % cmp(c, e))
e2 = grads(m, False)
print('UNPATCHED eager vs eager (determinism): conv1 %.2e worst %.2e' % cmp(e2, e))
import flash_rocm_kernels as K; K.install()
pe = grads(m, False); pc = grads(m, True)
print('PATCHED eager vs eager: conv1 %.2e worst %.2e' % cmp(pe, e))
print('PATCHED compiled vs UNPATCHED compiled: conv1 %.2e worst %.2e' % cmp(pc, c))
