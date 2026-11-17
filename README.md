# Flash ALOHA ACT on AMD Strix Halo (gfx1151): ROCm setup, YAM data, faster kernels

Working notes and code for training [Flash ALOHA](https://github.com/JohnYao93/flash-aloha) ACT policies on a
Radeon 8060S (gfx1151, ROCm nightly torch 2.10) with our own YAM robot recordings.

## Result

Batch 128, 3 cameras 180x320, ResNet18 + 7-layer decoder, 3000 steps, our YAM data.

| | wall time | speedup | val loss @3000 |
|---|---|---|---|
| stock | 58:46 | 1x | 0.0611 |
| kernels only (same math) | 44:55 | 1.31x | 0.0610 |
| + frozen backbones | 31:38 | 1.86x | 0.0566 |
| **+ GPU augmentation + CPU split** | **29:28** | **1.99x** | **0.0569** |

The kernel row alone is identical math (GPU step 1.12 s -> 0.765 s, 1.47x). The rest are opt-in recipe/pipeline
switches, described below. Val loss is offline; robot success is not measured here.

Validation curves (every 500 steps) are in `logs/val-curves.txt`; all benchmark rows in `logs/bench.jsonl`.

## What changed (all in `rocm/kernels/`)

Convolution weight/input gradients were the bottleneck: MIOpen/CK ran them at 0.5–10 TFLOPs on gfx1151.

| kernel | replaces | speedup vs MIOpen (batch 128) |
|---|---|---|
| `stem_wgrad.py` — implicit-GEMM wgrad for the 3->64 7x7/2 stem | MIOpen wgrad 66 ms/camera | ~7x |
| `conv_wgrad_s1.wgrad3x3` — pad-free 3x3/s1 wgrad, one program per kernel row (gy tile reused 3x), fp32 accumulation | CK wgrad | layer1 4.2x, layer2 3.2x, layer3 2.3x, layer4 1.4x |
| `conv_wgrad_s1.pointwise_wgrad` — 1x1/s2 downsample wgrad as one GEMM | MIOpen | 2–5x |
| input gradient of 3x3/s1 convs as a forward conv with flipped weights; `wflip.py` Triton flip (0.03 ms vs 0.73 ms) | CK bwd-data | 1.2–1.8x |
| TunableOp (`rocm/tuning/tunableop_results0.csv`) | default hipBLASLt/rocBLAS picks | ~5% of the step |

`flash_rocm_kernels.py` registers these as `torch.library` custom ops (opaque to `torch.compile`) and patches
`nn.Conv2d.forward` for eligible bf16 channels-last convs. Each op prints `RAN ...` once per shape at runtime as proof
it is active. Gradients vs an fp32 reference (pretrained weights, frozen BN): equal to or better than stock
(`test_grad_vs_fp32.py`).

Measured and left off: stride-2 3x3 wgrad (`conv_wgrad_s1.wgrad3x3_s2`; 1.2–4.2x per layer, but no full-step gain:
v5 0.757 s vs v7 0.770 s, `logs/kernel-sweep7.log`), Triton forward 3x3 conv (`conv3x3_fwd.py`, `FLASH_ROCM_FWD3X3=1`; ties CK inside the real
step), short-sequence Triton attention (`attn_short.py`; correct, backward slower than AOTriton), MIOpen exhaustive
find (no gain), `max-autotune` (slower). The transformer GEMMs already run at 24–34 TFLOPs.

## Opt-in switches beyond the kernels

- `FLASH_FREEZE_BACKBONE=1` — sets `requires_grad=False` on the ImageNet ResNet18 backbones (33.5M params) right
  after Flash loads them. Trained backbones moved only ~1% from ImageNet; frozen trains faster and ended with lower
  val loss here. This changes the recipe.
- `FLASH_GPU_AUG=1` — needs `rocm/patches/flash-gpu-aug.patch` applied to the Flash clone
  (`cd repo && git apply ../rocm/patches/flash-gpu-aug.patch`). The dataset ships uint8 frames; one fused Triton
  kernel does the exact /255 (lookup table, bit-identical to the CPU) plus the same 95% random crop + bilinear resize
  on the GPU (1.7 ms per batch; max diff 1.2e-7 vs torchvision). Validation is unchanged.
- `FLASH_CPU_SPLIT=1` — pins the training process to the last 2 allowed CPUs and dataloader workers to the rest, so
  workers don't starve the main thread.

## Use

```bash
# once: ROCm venv in repo/.venv-rocm (see rocm/overrides.txt, rocm/excludes.txt), the opt-in hook, the Flash patch
rocm/install/install_pth.sh
(cd repo && git apply ../rocm/patches/flash-gpu-aug.patch)
# every run
source rocm/env.sh
export FLASH_ROCM_KERNELS=1 PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 \
       PYTORCH_TUNABLEOP_FILENAME=$PWD/rocm/tuning/tunableop_results%d.csv
export FLASH_FREEZE_BACKBONE=1 FLASH_GPU_AUG=1 FLASH_CPU_SPLIT=1   # the 2x recipe; omit for same-math kernels only
cd repo && python scripts/run.py train ../yam-config/training/yam_h3_act.yaml --model-name <name>
```

Benchmark: `python ../tools/bench_act_step.py --training-config ../yam-config/training/yam_h3_act.yaml --model-name bench_x --steps 100 --profile --tag x`
(`--fresh` pulls a new batch every step and records loader wait).

## Data

`tools/convert_h3_recordings.py` and `tools/convert_teaching_demos.py` convert our YAM recordings (read-only on the
sources) to Flash episodes: 10 Hz, 14-D [left q1..6, jaw, right q1..6, jaw], cameras top/left/right. The data
itself is not in this repo.

## Gotchas

- Never load torch from a `.pth` without the bin/ guard: `offload-arch` in the ROCm venv is a Python script; an
  unguarded hook recursed into ~2.4k processes.
- An env-var re-entry guard breaks Flash: `scripts/run.py` re-execs itself, so the training process skipped the patch
  (a "kernels" run that was really stock — `logs/val-curves.txt`, INVALID-stock).
- `ulimit -u` counts threads, not processes.
- Microbenchmarks against MIOpen can mislead (MIOpen picks different kernels standalone vs in the compiled step);
  decide with the full step.
