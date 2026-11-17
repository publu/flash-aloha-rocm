#!/usr/bin/env bash
# fresh baseline after reboot: stock vs v5+TunableOp, interleaved x2
set -u
cd "$HOME/research/flash-aloha/repo"; source ../rocm/env.sh; export MIOPEN_LOG_LEVEL=1
OUT=../logs/bench.jsonl; CFG=../yam-config/training/yam_h3_act.yaml
export PYTORCH_TUNABLEOP_FILENAME=../rocm/tuning/tunableop_results%d.csv
run() { local tag=$1; shift; echo "=== $tag $(date +%T)"; rm -rf ../ckpts/bench_$tag
  timeout 2400 nice -n 19 taskset -c 24-31 python -u ../tools/bench_act_step.py --training-config $CFG --model-name bench_$tag \
     --steps 100 --warmup 15 --tag "$tag" --out $OUT "$@" > ../logs/bench-$tag.log 2>&1
  grep -h "BENCH" ../logs/bench-$tag.log | cut -c1-110 || tail -3 ../logs/bench-$tag.log; rm -rf ../ckpts/bench_$tag; }
for r in 1 2; do
  run base8-$r
  FLASH_ROCM_KERNELS=1 PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 run v5tun8-$r
done
echo "=== done $(date +%T)"
