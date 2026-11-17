# Flash ALOHA ACT on AMD Strix Halo (gfx1151): ROCm setup, YAM data, faster kernels

Working notes and code for training [Flash ALOHA](https://github.com/JohnYao93/flash-aloha) ACT policies on a
Radeon 8060S (gfx1151, ROCm nightly torch 2.10) with our own YAM robot recordings.

## Result

Same model, same recipe, same math — only kernels changed. Batch 128, 3 cameras 180x320, ResNet18 + 7-layer decoder.

| | stock | now | |
|---|---|---|---|
| GPU training step (`tools/bench_act_step.py`) | 1.12 s | **0.765 s** | **1.47x** |
| real 3000-step training (incl. validation) | 58:46 | **44:55** | **1.31x** |
| final val loss @3000 | 0.0611 | **0.0610** | same |

Validation curves (every 500 steps) are in `logs/val-curves.txt`; all benchmark rows in `logs/bench.jsonl`.

## What changed (all in `rocm/kernels/`)

Convolution weight/input gradients were the bottleneck: MIOpen/CK ran them at 0.5–10 TFLOPs on gfx1151.

| kernel | replaces | speedup vs MIOpen (batch 128) |
|---|---|---|
| `stem_wgrad.py` — implicit-GEMM wgrad for the 3->64 7x7/2 stem | MIOpen wgrad 66 ms/camera | ~7x |
| `conv_wgrad_s1.wgrad3x3` — pad-free 3x3/s1 wgrad, one program per kernel row (gy tile reused 3x), fp32 accumulation | CK wgrad | layer1 4.2x, layer2 3.2x, layer3 2.3x, layer4 1.4x |
| `conv_wgrad_s1.wgrad3x3_s2` — same for 3x3/s2 (layerN.0.conv1) | CK wgrad | 4.2x / 1.2x / 1.3x |
| `conv_wgrad_s1.pointwise_wgrad` — 1x1/s2 downsample wgrad as one GEMM | MIOpen | 2–5x |
| input gradient of 3x3/s1 convs as a forward conv with flipped weights; `wflip.py` Triton flip (0.03 ms vs 0.73 ms) | CK bwd-data | 1.2–1.8x |
| TunableOp (`rocm/tuning/tunableop_results0.csv`) | default hipBLASLt/rocBLAS picks | ~5% of the step |

`flash_rocm_kernels.py` registers these as `torch.library` custom ops (opaque to `torch.compile`) and patches
`nn.Conv2d.forward` for eligible bf16 channels-last convs. Each op prints `RAN ...` once per shape at runtime as proof
it is active. Gradients vs an fp32 reference (pretrained weights, frozen BN): equal to or better than stock
(`test_grad_vs_fp32.py`).

Measured and left off: Triton forward 3x3 conv (`conv3x3_fwd.py`, `FLASH_ROCM_FWD3X3=1`; ties CK inside the real
step), short-sequence Triton attention (`attn_short.py`; correct, backward slower than AOTriton), MIOpen exhaustive
find (no gain), `max-autotune` (slower). The transformer GEMMs already run at 24–34 TFLOPs.

## Use

```bash
# once: ROCm venv in repo/.venv-rocm (see rocm/overrides.txt, rocm/excludes.txt), then the opt-in hook
rocm/install/install_pth.sh
# every run
source rocm/env.sh
export FLASH_ROCM_KERNELS=1 PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 \
       PYTORCH_TUNABLEOP_FILENAME=$PWD/rocm/tuning/tunableop_results%d.csv
cd repo && python scripts/run.py train ../yam-config/training/yam_h3_act.yaml --model-name <name>
```

Benchmark: `python ../tools/bench_act_step.py --training-config ../yam-config/training/yam_h3_act.yaml --model-name bench_x --steps 100 --profile --tag x`.

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
