"""Steady-state throughput of Flash's train dataloader alone (no GPU work), same config/workers as training."""
import os, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1] / 'repo'
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
import policy.imitate_episodes_train as T
N = int(os.environ.get('NB', 60))
def bench(train_dataloader, val_dataloader, config, stats=None):
    it = T.repeater(train_dataloader)
    for _ in range(10): next(it)                      # warm: workers spawned, prefetch filled
    t0 = time.perf_counter()
    for _ in range(N): b = next(it)
    dt = (time.perf_counter() - t0) / N
    print(f'DATA steady {dt:.3f} s/batch = {1/dt:.2f} batch/s  workers={config.get("num_dataloader_workers")} '
          f'shapes={[tuple(x.shape) for x in b]} dtypes={[str(x.dtype) for x in b]}', flush=True)
    os._exit(0)
if __name__ == '__main__':
    T.train_bc = bench
    sys.argv = [str(REPO / 'scripts/policy/imitate_episodes_train.py')] + sys.argv[1:]
    sys.exit(T.main())
