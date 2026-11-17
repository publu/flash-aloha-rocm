import torch, torchvision
import flash_rocm_kernels as K, stem_wgrad as S
calls = {'n': 0}; orig = S.stem_wgrad
def counted(*a, **k): calls['n'] += 1; return orig(*a, **k)
S.stem_wgrad = counted
K.install()
import flash_rocm_kernels
m = torchvision.models.resnet18(weights=None).cuda().to(torch.bfloat16).to(memory_format=torch.channels_last)
x = torch.randn(8, 3, 180, 320, device='cuda', dtype=torch.bfloat16).contiguous(memory_format=torch.channels_last)
m(x).float().mean().backward(); print('eager wgrad calls', calls['n'])
f = torch.compile(m); f(x).float().mean().backward(); print('compiled wgrad calls (cumulative)', calls['n'])
import torch._dynamo as D
print(D.explain(m)(x).break_reasons[:3])
