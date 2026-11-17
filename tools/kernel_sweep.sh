#!/usr/bin/env bash
# Kernel-only sweep for Flash ACT training on gfx1151. Same model/recipe/math; only kernel selection changes.
# Each tuned variant runs twice: pass 1 tunes + caches, pass 2 measures with the cache.
set -u
cd "$HOME/research/flash-aloha/repo"
source ../rocm/env.sh
export MIOPEN_LOG_LEVEL=1
OUT=../logs/bench.jsonl
CFG=../yam-config/training/yam_h3_act.yaml
T=../rocm/tuning; mkdir -p "$T"
run() { # tag, extra flash args...
  local tag=$1; shift
  echo "=== $tag $(date +%T)"
  rm -rf ../ckpts/bench_$tag
  nice -n 19 taskset -c 24-31 python -u ../tools/bench_act_step.py --training-config $CFG --model-name bench_$tag \
     --steps "${STEPS:-100}" --warmup 15 --profile --tag "$tag" --out $OUT "$@" > ../logs/bench-$tag.log 2>&1
  grep -hE "TIMING|BENCH" ../logs/bench-$tag.log | cut -c1-400 || tail -3 ../logs/bench-$tag.log
  rm -rf ../ckpts/bench_$tag
}
# 0. baseline exactly as Flash ships it (compile default) and without compile
run base
run nocompile --no-compile
# 1. TunableOp: benchmark rocBLAS + hipBLASLt GEMM solutions per shape, cache results
export PYTORCH_TUNABLEOP_FILENAME=$T/tunableop_results%d.csv
PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=1 run tunableop-tune
PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 run tunableop
# 2. MIOpen exhaustive conv search into a private user DB, then measure from the DB
export MIOPEN_USER_DB_PATH=$T/miopen; export MIOPEN_CUSTOM_CACHE_DIR=$T/miopen-cache; mkdir -p $MIOPEN_USER_DB_PATH $MIOPEN_CUSTOM_CACHE_DIR
MIOPEN_FIND_MODE=1 MIOPEN_FIND_ENFORCE=3 run miopen-tune
MIOPEN_FIND_MODE=1 run miopen
# 3. both
PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 MIOPEN_FIND_MODE=1 run tunableop+miopen
# 4. compiler autotune on top
PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0 MIOPEN_FIND_MODE=1 run all+max-autotune --compile_mode max-autotune
run base-end
echo "=== done $(date +%T)"
