"""Benchmark Flash ALOHA's exact ACT training step on real YAM batches (kernel-level tuning only).

Runs Flash's own trainer setup (same policy, optimizer, compile, channels-last), but swaps train_bc for a timing loop:
  1. data:    time to pull --data-batches real batches from Flash's train dataloader
  2. compute: --steps fwd+bwd+AdamW steps on those real batches (resident on GPU, cycled), after --warmup steps
  3. optional --profile: kernel-time breakdown by category
Also records the loss on a fixed held batch so kernel changes can be checked for numerical drift.

usage (from repo/, after `source ../rocm/env.sh`):
  python ../tools/bench_act_step.py --training-config ../yam-config/training/yam_h3_act.yaml [--steps 100] [--profile] [--tag NAME]
Extra Flash training flags (e.g. --compile_mode max-autotune, --no-compile) pass through.
"""
import json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1] / 'repo'
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]

own = {'--steps': 100, '--warmup': 15, '--data-batches': 12, '--tag': 'run', '--out': str(REPO.parent / 'logs' / 'bench.jsonl')}
flags = {'--profile': False, '--fresh': False}
argv, rest = sys.argv[1:], []
i = 0
while i < len(argv):
    a = argv[i]
    if a in own:
        own[a] = type(own[a])(argv[i + 1]) if not isinstance(own[a], str) else argv[i + 1]; i += 2
    elif a in flags:
        flags[a] = True; i += 1
    else:
        rest.append(a); i += 1

import torch
import policy.imitate_episodes_train as T  # noqa: E402


def bench(train_dataloader, val_dataloader, config, stats=None):
    from policy.model_util import make_policy, make_optimizer  # same factories the trainer uses
    T.set_seed(config['seed'])
    policy = make_policy(config['policy_class'], config['policy_config'])
    policy.load_pretrained_backbone()
    policy.cuda()
    policy.model.use_channels_last()
    if config.get('compile'):
        import torch._inductor.config as ic
        ic.coordinate_descent_tuning = bool(config.get('compile_coordinate_descent'))
        policy._compute_loss = torch.compile(policy._compute_loss, mode=config.get('compile_mode', 'max-autotune'))
    optimizer = make_optimizer(config['policy_class'], policy)

    it = T.repeater(train_dataloader)
    t0 = time.perf_counter()
    batches = [next(it) for _ in range(int(own['--data-batches']))]
    data_s = (time.perf_counter() - t0) / len(batches)
    batches = [tuple(t.cuda(non_blocking=True) for t in b) for b in batches]
    held = batches[0]

    def step(b):
        policy.train(); optimizer.zero_grad()
        d = T.forward_pass(b, policy, 0)
        d['loss'].backward(); optimizer.step()
        return d['loss']

    t0 = time.perf_counter()
    for k in range(int(own['--warmup'])):
        step(batches[k % len(batches)])
    torch.cuda.synchronize(); warm_s = time.perf_counter() - t0

    n = int(own['--steps'])
    wait_s = [0.0]
    t0 = time.perf_counter()
    for k in range(n):
        if flags['--fresh']:  # like the real loop: a new batch from the dataloader every step, copied to GPU
            tw = time.perf_counter(); b = next(it); wait_s[0] += time.perf_counter() - tw
            step(tuple(t.cuda(non_blocking=True) for t in b))
        else:
            step(batches[k % len(batches)])
    torch.cuda.synchronize(); step_s = (time.perf_counter() - t0) / n

    prof_rows = None
    if flags['--profile']:
      try:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            for k in range(10):
                step(tuple(t.cuda(non_blocking=True) for t in next(it)) if flags['--fresh'] else batches[k % len(batches)])
            torch.cuda.synchronize()
        cats = {}
        tot = 0.0
        for e in p.key_averages():
            dt = getattr(e, 'self_device_time_total', 0) or getattr(e, 'self_cuda_time_total', 0)
            if dt <= 0:
                continue
            nm = e.key.lower(); tot += dt
            c = ('conv' if any(s in nm for s in ('conv', 'miopen', 'igemm', 'winograd', 'implicit')) else
                 'gemm' if any(s in nm for s in ('gemm', 'cijk', 'matmul', 'hipblaslt', 'rocblas', 'mm_')) else
                 'attention' if any(s in nm for s in ('attn', 'attention', 'fmha', 'flash')) else
                 'norm' if 'norm' in nm else
                 'optimizer' if any(s in nm for s in ('adam', 'foreach', 'multi_tensor')) else
                 'triton-fused' if 'triton' in nm else
                 'copy/layout' if any(s in nm for s in ('copy', 'transpose', 'memcpy', 'cat', 'contiguous')) else 'other')
            cats[c] = cats.get(c, 0) + dt
        prof_rows = {k: round(100 * v / tot, 1) for k, v in sorted(cats.items(), key=lambda x: -x[1])}
        prof_rows['_gpu_ms_per_step'] = round(tot / 10 / 1000, 1)
        top = sorted(p.key_averages(), key=lambda e: -(getattr(e, 'self_device_time_total', 0) or 0))[:12]
        from torch.autograd import DeviceType
        kern = [e for e in p.key_averages() if getattr(e, 'device_type', None) == DeviceType.CUDA]
        ktot = sum(e.self_device_time_total for e in kern)
        prof_rows['_kernel_ms_per_step'] = round(ktot / 10 / 1000, 1)
        prof_rows['_kernels'] = [(e.key[:90], round(e.self_device_time_total / 10 / 1000, 2), e.count // 10)
                                 for e in sorted(kern, key=lambda e: -e.self_device_time_total)[:400]]
        prof_rows['_top'] = [(e.key[:70], round((getattr(e, 'self_device_time_total', 0) or 0) / 10 / 1000, 2)) for e in top]
      except Exception as exc:
        prof_rows = {'error': repr(exc)[:300]}

    print('TIMING', json.dumps(dict(tag=own['--tag'], step_s=step_s, data_s=data_s, prof=prof_rows)), flush=True)
    # held-batch loss in eval mode (no dropout), for numerical comparison across kernel settings
    with torch.inference_mode():
        policy.eval(); T.set_seed(0)
        held_loss = float(T.forward_pass(held, policy, 0)['loss'].float())

    res = dict(tag=own['--tag'], step_s=round(step_s, 4), samples_per_s=round(int(held[0].shape[0]) / step_s, 1), batch=int(held[0].shape[0]),
               data_s_per_batch=round(data_s, 4), loader_wait_s=round(wait_s[0] / n, 4), warmup_s=round(warm_s, 1), held_loss_after=round(held_loss, 5),
               env={k: v for k, v in os.environ.items() if k.startswith(('PYTORCH_TUNABLEOP', 'MIOPEN', 'TORCH_ROCM', 'HIPBLASLT', 'TORCH_BLAS'))},
               compile=config.get('compile'), compile_mode=config.get('compile_mode'), profile=prof_rows)
    print('BENCH', json.dumps(res), flush=True)
    with open(own['--out'], 'a') as f:
        f.write(json.dumps(res) + '\n')
    os._exit(0)


if __name__ == '__main__':  # dataloader workers re-import this file; only the parent runs the benchmark
    T.train_bc = bench
    sys.argv = [str(REPO / 'scripts/policy/imitate_episodes_train.py')] + rest
    sys.exit(T.main())
