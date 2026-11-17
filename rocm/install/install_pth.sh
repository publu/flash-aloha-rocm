#!/usr/bin/env bash
# Installs the opt-in startup hook into the ROCm venv: with FLASH_ROCM_KERNELS=1 (or =stem), every Python process of
# the venv patches torch.nn.Conv2d with the Triton kernels. Skips scripts living in the venv bin/ (ROCm tool wrappers
# such as offload-arch are Python scripts there; loading torch in them recursed into a fork bomb).
set -eu
HERE="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${1:-$HERE/../repo/.venv-rocm}"
SP=$("$VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')
cat > "$SP/flash_rocm_kernels.pth" <<PTH
import os, sys; (os.environ.get('FLASH_ROCM_KERNELS') in ('1', 'stem')) and not any(os.path.dirname(os.path.abspath(a)) == os.path.join(sys.prefix, 'bin') for a in sys.orig_argv[1:2]) and (sys.path.insert(0, '$HERE/kernels') or __import__('flash_rocm_kernels'))
PTH
echo "installed $SP/flash_rocm_kernels.pth -> $HERE/kernels"
