#!/usr/bin/env bash
# stock vs v5 (validated) vs v7 (v5 + stride-2 3x3 wgrad), all with TunableOp, interleaved x2
set -u
cd "$HOME/research/flash-aloha/repo"; source ../rocm/env.sh; export MIOPEN_LOG_LEVEL=1
OUT=../logs/bench.jsonl; CFG=../yam-config/training/yam_h3_act.yaml; T=../rocm/tuning
export PYTORCH_TUNABLEOP_FILENAME=$T/tunableop_results%d.csv PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0
K=../rocm/kernels; V5=${V5:?set V5 to the v5 flash_rocm_kernels.py to compare against}
run() { local tag=$1; shift; echo "=== $tag $(date +%T)"; rm -rf ../ckpts/bench_$tag
  timeout 2400 nice -n 19 taskset -c 24-31 python -u ../tools/bench_act_step.py --training-config $CFG --model-name bench_$tag \
     --steps 100 --warmup 15 --profile --tag "$tag" --out $OUT "$@" > ../logs/bench-$tag.log 2>&1
  grep -h "BENCH" ../logs/bench-$tag.log | cut -c1-160 || tail -3 ../logs/bench-$tag.log; rm -rf ../ckpts/bench_$tag; }
cp $K/flash_rocm_kernels.py $K/flash_rocm_kernels.v7.py
for r in 1 2; do
  run tbase7-$r
  cp $V5 $K/flash_rocm_kernels.py; FLASH_ROCM_KERNELS=1 run tv5-$r
  cp $K/flash_rocm_kernels.v7.py $K/flash_rocm_kernels.py; FLASH_ROCM_KERNELS=1 run tv7-$r
done
cp $K/flash_rocm_kernels.v7.py $K/flash_rocm_kernels.py
echo "=== done $(date +%T)"
