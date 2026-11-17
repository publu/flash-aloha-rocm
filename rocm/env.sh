# Source before running Flash ALOHA on the Strix Halo iGPU (gfx1151, ROCm 7.13 / torch 2.10 TheRock wheels in repo/.venv-rocm).
export FLASH_ALOHA_VENV=$HOME/research/flash-aloha/repo/.venv-rocm
export PATH=$FLASH_ALOHA_VENV/bin:$PATH
# AOTriton flash / mem-efficient SDPA are gated as "experimental" on gfx1151; measured 4-6x faster than MATH, same results to bf16 rounding.
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export FLASH_ALOHA_TASK_DIRS=$HOME/research/flash-aloha/yam-config/tasks
export FLASH_ALOHA_DATA_DIR=$HOME/research/flash-aloha/data
export FLASH_ALOHA_CKPT_DIR=$HOME/research/flash-aloha/ckpts
export FLASH_ALOHA_LOG_DIR=$HOME/research/flash-aloha/logs
export WANDB_MODE=disabled
