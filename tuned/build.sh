#!/bin/bash
# tuned/build.sh <variant> — build the tuned sglang-kernel and sglang wheels for one GPU variant (gb10) and one
# CUDA toolkit. It creates the toolkit's build venv (tuned/build/<cuda-tag>/venv), installs the build and runtime
# dependencies into it from tuned/requirements-<variant>.txt, then makes the unstamped wheels with tuned/wheel.sh.
# The sglang-kernel compile is the long step. Nothing is installed into the user site or ~/.local.
#
#   CUDA_TAG=cu134 bash tuned/build.sh gb10
#
# Stage order (tuned-common docs/ARCHITECTURE.md B3): build.sh -> full_test.sh -> package.sh -> smoke_test.sh ->
# (review and commit the release notes) -> release.sh -> publish.sh.
set -euo pipefail

GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/build.sh <variant>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=env.sh
source "${REPO_ROOT}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}"

command -v python3 &>/dev/null || { echo "ERROR: python3 not found on PATH." >&2; exit 1; }
[[ -f "${GPU_TUNED_REQUIREMENTS}" ]] || { echo "ERROR: ${GPU_TUNED_REQUIREMENTS} (GPU_TUNED_REQUIREMENTS in tuned/repo.env) not found." >&2; exit 1; }

VENV_DIR="${GPU_TUNED_VENV_DIR}"
# Create or reuse (verified), activate, upgrade pip, current setuptools: one shared setup (C9, tuned-common).
gpu_tuned_setup_build_venv "${VENV_DIR}" "${REPO_ROOT}" || exit 1

echo "=========================================="
echo "Building sglang for ${GPU_TUNED_HW_LABEL}"
echo "=========================================="
echo "Toolkit:   ${CUDA_TAG} (${CUDA_HOME})"
echo "Venv:      ${VENV_DIR}"
echo "Python:    $(python3 --version)"
echo ""

echo "Installing build-time requirements..."
# sglang-kernel: scikit-build-core (pyproject build-system), a CMake newer than the host's, ninja.
# sglang: setuptools (below) and setuptools-scm (its version comes from git).
pip install build wheel "scikit-build-core>=0.10" "cmake>=4.0" ninja "setuptools-scm>=8"

echo "Installing ${GPU_TUNED_VARIANT} requirements (torch @ URL, flashinfer)"
pip install -r "${GPU_TUNED_REQUIREMENTS}"
# Guard the tuned torch: anything below that pins another torch now fails to resolve instead of replacing it.
TORCH_VERSION="$(python3 -c 'import torch; print(torch.__version__)')"
gpu_tuned_protect_torch_pin "${VENV_DIR}" "${TORCH_VERSION}" || exit 1

echo "Installing sglang's other dependencies (upstream python/pyproject.toml minus GPU_TUNED_DROP_DEPS)"
# shellcheck disable=SC2086 # GPU_TUNED_DROP_DEPS is a word list
python3 -I tuned/deps.py upstream python/pyproject.toml ${GPU_TUNED_DROP_DEPS} > "${GPU_TUNED_BUILD_ROOT}/upstream-deps.txt"
pip install -r "${GPU_TUNED_BUILD_ROOT}/upstream-deps.txt"

echo "Auditing already-installed tuned packages"
# Non-fatal: surfaces a stock torch or flashinfer that replaced a tuned one before the build relies on it.
gpu_tuned_audit_pinned pip "torch=${GPU_TUNED_VARIANT}" "flashinfer-python=${GPU_TUNED_VARIANT}"
gpu_tuned_audit_stray pip flashinfer-cubin

echo "Validating the tuned torch's build-info stamp"
# Fatal on purpose: sglang-kernel links this torch, so a stock torch here means a wrong build.
TORCH_CUDA_SO="$(python3 -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib", "libtorch_cuda-tuned.so"))')"
gpu_tuned_verify_build_info_json "${TORCH_CUDA_SO}" "variant=${GPU_TUNED_VARIANT}" "package=torch" || exit 1

# The current setuptools, installed last (C8, tuned-common): a requirement above may have pulled it back down.
gpu_tuned_install_setuptools || exit 1

echo ""
bash "${REPO_ROOT}/tuned/wheel.sh" "${GPU_TUNED_VARIANT}"

# Seed sglang's runtime JIT cache from the acceptance models (the shipped wheel carries it; package.sh), then
# prove it: the same models in strict mode, where any nvcc compile at run time is an error.
echo ""
echo "Seeding the sglang JIT cache (${SGLANG_JIT_CACHE_DIR}) from the acceptance models"
rm -f "${GPU_TUNED_BUILD_ROOT}/jit-cache-seeded"
rm -rf "${SGLANG_JIT_CACHE_DIR:?}"
mkdir -p "${SGLANG_JIT_CACHE_DIR}"
pip install --no-deps --force-reinstall "${GPU_TUNED_WHEEL_DIR}"/sglang_kernel-*.whl "${GPU_TUNED_WHEEL_DIR}"/sglang-*.whl
export CUDA_VISIBLE_DEVICES="${GPU_TUNED_TEST_GPUS:-0}"
(cd "${GPU_TUNED_BUILD_ROOT}" && python3 -I "${REPO_ROOT}/tuned/gb10_acceptance.py" seed) || { echo "ERROR: seeding the JIT cache failed." >&2; exit 1; }
(cd "${GPU_TUNED_BUILD_ROOT}" && python3 -I "${REPO_ROOT}/tuned/gb10_acceptance.py" strict) || { echo "ERROR: the seeded JIT cache does not cover the acceptance models (strict mode)." >&2; exit 1; }
SEEDED="$(find "${SGLANG_JIT_CACHE_DIR}" -name '*.so' | wc -l)"
echo "OK: ${SEEDED} JIT modules seeded; strict mode passed"
cat "${GPU_TUNED_BUILD_ROOT}/release-label" > "${GPU_TUNED_BUILD_ROOT}/jit-cache-seeded"

echo ""
echo "=========================================="
echo "Build complete (${GPU_TUNED_HW_LABEL})."
echo "=========================================="
echo "Label:        $(cat "${GPU_TUNED_BUILD_ROOT}/release-label")"
echo "Build commit: $(cat "${GPU_TUNED_BUILD_ROOT}/build-commit")"
echo "Next: CUDA_TAG=${CUDA_TAG} bash tuned/full_test.sh ${GPU_TUNED_VARIANT}"
