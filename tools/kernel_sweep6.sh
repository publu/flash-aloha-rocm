#!/usr/bin/env bash
# A/B/C interleaved: stock vs v6 (v5 + Triton fwd 3x3 for forward and dgrad) vs v6 + TunableOp
set -u
cd "$HOME/research/flash-aloha/repo"; source ../rocm/env.sh; export MIOPEN_LOG_LEVEL=1
OUT=../logs/bench.jsonl; CFG=../yam-config/training/yam_h3_act.yaml; T=../rocm/tuning
export PYTORCH_TUNABLEOP_FILENAME=$T/tunableop_results%d.csv
run() { local tag=$1; shift; echo "=== $tag $(date +%T)"; rm -rf ../ckpts/bench_$tag
  timeout 2400 nice -n 19 taskset -c 24-31 python -u ../tools/bench_act_step.py --training-config $CFG --model-name bench_$tag \
     --steps 100 --warmup 15 --profile --tag "$tag" --out $OUT "$@" > ../logs/bench-$tag.log 2>&1
  grep -h "BENCH" ../logs/bench-$tag.log | cut -c1-200 || tail -3 ../logs/bench-$tag.log; rm -rf ../ckpts/bench_$tag; }
for r in 1 2; do
  run base6-$r
  FLASH_ROCM_KERNELS=1 run v6-$r
  FLASH_ROCM_KERNELS=1 PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 run v6tun-$r
done
echo "=== done $(date +%T)"
