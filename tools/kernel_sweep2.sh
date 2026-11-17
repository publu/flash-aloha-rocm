#!/usr/bin/env bash
set -u
cd "$HOME/research/flash-aloha/repo"; source ../rocm/env.sh; export MIOPEN_LOG_LEVEL=1
OUT=../logs/bench.jsonl; CFG=../yam-config/training/yam_h3_act.yaml; T=../rocm/tuning
run() { local tag=$1; shift; echo "=== $tag $(date +%T)"; rm -rf ../ckpts/bench_$tag
  nice -n 19 taskset -c 24-31 python -u ../tools/bench_act_step.py --training-config $CFG --model-name bench_$tag \
     --steps 100 --warmup 15 --profile --tag "$tag" --out $OUT "$@" > ../logs/bench-$tag.log 2>&1
  grep -hE "BENCH" ../logs/bench-$tag.log | cut -c1-300 || tail -3 ../logs/bench-$tag.log; rm -rf ../ckpts/bench_$tag; }
export PYTORCH_TUNABLEOP_FILENAME=$T/tunableop_results%d.csv
run base2
FLASH_ROCM_KERNELS=1 run stemkernel
FLASH_ROCM_KERNELS=1 PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 run stemkernel+tunableop
run base2-end
echo "=== done $(date +%T)"
