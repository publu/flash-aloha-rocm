"""attn_short vs fp32 reference (same dropout mask, made explicit) and vs PyTorch SDPA backends; timing interleaved.
Layout as nn.MultiheadAttention passes it: (L, B*H, D) memory viewed as (B, H, L, D)."""
import time, itertools, torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from attn_short import attn_fwd, attn_bwd, dropout_mask
torch.manual_seed(0)
B, H, L, D, P = 128, 8, 201, 64, 0.1
def mk():  # MHA layout: (L, B*H, D) -> view (B*H, L, D) -> (B, H, L, D)
    t = torch.randn(L, B * H, D, device='cuda', dtype=torch.bfloat16)
    return t.transpose(0, 1).view(B, H, L, D)
q, k, v, do = mk(), mk(), mk(), mk()
seed = 1234
# reference in fp32 with the identical mask
M = dropout_mask(B, H, L, P, seed)
qf, kf, vf = (t.float().requires_grad_() for t in (q, k, v))
Pm = torch.softmax(qf @ kf.transpose(-1, -2) * D ** -0.5, -1)
of = (Pm * M / (1 - P)) @ vf
of.backward(do.float())
o, lse = attn_fwd(q, k, v, P, seed)
dq, dk, dv = attn_bwd(do, q, k, v, o, lse, P, seed)
rel = lambda a, b: ((a.float() - b).norm() / b.norm()).item()
print(f'vs fp32 (same mask): out {rel(o, of):.1e}  dq {rel(dq, qf.grad):.1e}  dk {rel(dk, kf.grad):.1e}  dv {rel(dv, vf.grad):.1e}')
# bf16 SDPA reference error at p=0 for scale
with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    q2, k2, v2 = (t.detach().clone().requires_grad_() for t in (q, k, v))
    o2 = F.scaled_dot_product_attention(q2, k2, v2); o2.backward(do)
qf2, kf2, vf2 = (t.float().requires_grad_() for t in (q, k, v))
o3 = torch.softmax(qf2 @ kf2.transpose(-1, -2) * D ** -0.5, -1) @ vf2; o3.backward(do.float())
o4, l4 = attn_fwd(q, k, v, 0.0, 0); g4 = attn_bwd(do, q, k, v, o4, l4, 0.0, 0)
print(f'p=0 vs fp32: SDPA-efficient out {rel(o2, o3):.1e} dq {rel(q2.grad, qf2.grad):.1e} | triton out {rel(o4, o3):.1e} dq {rel(g4[0], qf2.grad):.1e} dk {rel(g4[1], kf2.grad):.1e} dv {rel(g4[2], vf2.grad):.1e}')
print('dropout keep fraction', M.float().mean().item())
def t(fn, n=20):
    fn(); torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - s) / n * 1e3
qs, ks_, vs = (t_.detach().clone().requires_grad_() for t_ in (q, k, v))
def sdpa(be):
    def f():
        with sdpa_kernel(be):
            o = F.scaled_dot_product_attention(qs, ks_, vs, dropout_p=P)
        o.backward(do)
    return f
fl = 4 * B * H * L * L * D * 3.5
res = []
for bm, bn, nw, ns in itertools.product((32, 64, 128), (32, 64, 128), (4, 8), (1, 2)):
    cfg = dict(BM=bm, BN=bn, num_warps=nw, num_stages=ns)
    def f():
        o, l = attn_fwd(q, k, v, P, seed, cfg=cfg); attn_bwd(do, q, k, v, o, l, P, seed, cfg=cfg)
    try: f()
    except Exception as e: continue
    res.append((t(f), cfg))
res.sort(key=lambda r: r[0])
fa = t(sdpa(SDPBackend.FLASH_ATTENTION)); ea = t(sdpa(SDPBackend.EFFICIENT_ATTENTION))
print(f'fwd+bwd dropout {P}: flash {fa:.2f} ms | efficient {ea:.2f} ms | triton best {res[0][0]:.2f} ms ({fl/res[0][0]/1e9:.1f} TF) {res[0][1]}  -> {fa/res[0][0]:.2f}x vs flash')
for r in res[1:4]: print('  ', round(r[0], 2), r[1])
