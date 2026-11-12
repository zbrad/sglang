#!/bin/bash
# tuned/env.sh <variant> — device, toolkit and build-environment setup for the tuned sglang wheels
# (sglang-kernel, built from python/sglang/kernels/aot, and sglang itself). Source this file with the
# variant as $1; do not execute it directly.
#
#   CUDA_TAG=cu134 bash tuned/build.sh gb10     (or CUDA_VER=13.4)
#
# Device facts come from tuned/common-data/devices/<variant>.conf and the CUDA toolkit from
# tuned/common-data/toolkits/<id>.env (both vendored from zbrad/tuned-common by sync-common.sh), never
# from /usr/local/cuda or from what happens to be installed. There is no default: CUDA_TAG or CUDA_VER
# must name one. Repo-specific settings live in tuned/repo.env, loaded after the shared definitions; it
# may not change any shared key, and it restricts the toolkits this repo builds against (gb10: cu134).
#
# Exported: GPU_TUNED_VARIANT/PLATFORM/CUDA_ARCH/HW_LABEL (shared), the repo.env keys, CUDA_HOME/CUDA_TAG/
# CUDA_VER (the toolkit), GPU_TUNED_BUILD_ROOT (tuned/build/<cuda-tag>, the one tree for this toolkit:
# venv/ wheel/ cmake/ dist/ releases/ ccache/ jit-cache/), GPU_TUNED_VENV_DIR, GPU_TUNED_WHEEL_DIR,
# GPU_TUNED_DIST_DIR, GPU_TUNED_RELEASES_DIR, GPU_TUNED_CMAKE_DIR, CCACHE_DIR, MAX_JOBS, NVCC_THREADS,
# CMAKE_BUILD_PARALLEL_LEVEL and SGLANG_JIT_CACHE_DIR.

GPU_TUNED_ARG_VARIANT="${1:-}"
if [[ -z "${GPU_TUNED_ARG_VARIANT}" ]]; then
    echo "ERROR: env.sh requires a variant argument (gb10)" >&2
    return 1 2>/dev/null || exit 1
fi

GPU_TUNED_SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_TUNED_DATA_DIR="${GPU_TUNED_SELF_DIR}/common-data"
if [[ ! -f "${GPU_TUNED_DATA_DIR}/devices/${GPU_TUNED_ARG_VARIANT}.conf" ]]; then
    echo "ERROR: no device '${GPU_TUNED_ARG_VARIANT}' in ${GPU_TUNED_DATA_DIR}/devices (have: $(cd "${GPU_TUNED_DATA_DIR}/devices" && ls | sed 's/\.conf$//' | tr '\n' ' '))" >&2
    return 1 2>/dev/null || exit 1
fi
# shellcheck source=common-data/devices/gb10.conf
source "${GPU_TUNED_DATA_DIR}/devices/${GPU_TUNED_ARG_VARIANT}.conf" || return 1 2>/dev/null || exit 1
export GPU_TUNED_VARIANT GPU_TUNED_PLATFORM GPU_TUNED_CUDA_ARCH GPU_TUNED_HW_LABEL

# shellcheck source=common.sh
# Vendored from https://github.com/zbrad/tuned-common (see sync-common.sh).
source "${GPU_TUNED_SELF_DIR}/common.sh" || return 1 2>/dev/null || exit 1

# Fail loudly if this script runs on the wrong host, rather than producing a wrong-architecture wheel.
gpu_tuned_assert_platform "${GPU_TUNED_PLATFORM}" "${GPU_TUNED_VARIANT}" || return 1 2>/dev/null || exit 1

# --- CUDA toolkit: CUDA_TAG (cu134) or CUDA_VER (13.4) name one from common-data/toolkits. ---
_sgl_tk="${CUDA_TAG:-}"
_sgl_req_ver="${CUDA_VER:-}"
if [ -z "${_sgl_tk}" ] && [ -n "${_sgl_req_ver}" ]; then
    _sgl_tk="cu${_sgl_req_ver//./}"                            # 13.4 -> cu134
fi
gpu_tuned_load_toolkit "${GPU_TUNED_DATA_DIR}" "${GPU_TUNED_VARIANT}" "${_sgl_tk}" || return 1 2>/dev/null || exit 1
if [ -n "${_sgl_req_ver}" ] && [ "${_sgl_req_ver}" != "${TOOLKIT_CUDA_VER}" ]; then
    echo "[tuned/env] ERROR: CUDA_VER=${_sgl_req_ver} conflicts with toolkit ${TOOLKIT_ID} (CUDA ${TOOLKIT_CUDA_VER})." >&2
    return 1 2>/dev/null || exit 1
fi
unset _sgl_tk _sgl_req_ver

# --- Repo-specific settings (loaded after the shared device and toolkit values; may not change them). ---
gpu_tuned_load_repo_env "${GPU_TUNED_SELF_DIR}/repo.env" || return 1 2>/dev/null || exit 1
# A sourced file's own `return 1` does not stop the loader, so check that repo.env really set its keys.
if [[ -z "${GPU_TUNED_REQUIREMENTS:-}" || -z "${GPU_TUNED_REPO_TOOLKITS:-}" ]]; then
    echo "ERROR: tuned/repo.env did not set its settings for ${GPU_TUNED_VARIANT}." >&2
    return 1 2>/dev/null || exit 1
fi
export GPU_TUNED_UPSTREAM_REF GPU_TUNED_FORK_MARKERS GPU_TUNED_REQUIREMENTS GPU_TUNED_DROP_DEPS
export GPU_TUNED_REPO_TOOLKITS GPU_TUNED_REPO_TOOLKIT_REASON

# Which toolkits this variant may be built against (repo.env). Checked before the toolkit is verified, so
# a refused toolkit is reported as the restriction, not as a missing install.
if [[ " ${GPU_TUNED_REPO_TOOLKITS} " != *" ${CUDA_TAG} "* ]]; then
    echo "[tuned/env] ERROR: ${GPU_TUNED_VARIANT} sglang builds support ${GPU_TUNED_REPO_TOOLKITS} only; ${CUDA_TAG} was asked for." >&2
    echo "  ${GPU_TUNED_REPO_TOOLKIT_REASON}" >&2
    return 1 2>/dev/null || exit 1
fi
gpu_tuned_verify_toolkit || return 1 2>/dev/null || exit 1

export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${GPU_TUNED_TOOLKIT_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_PATH="${CUDA_HOME}" CUDACXX="${CUDA_HOME}/bin/nvcc"
CUDA_VERSION_COMPACT="${CUDA_TAG#cu}"
export CUDA_VERSION_COMPACT

# --- The one build tree for this toolkit (docs/ARCHITECTURE.md C1/C2): tuned/build/<cuda-tag>/. ---
GPU_TUNED_REPO_ROOT="$(cd "${GPU_TUNED_SELF_DIR}/.." && pwd)"
gpu_tuned_claim_toolkit_tree "${GPU_TUNED_REPO_ROOT}" "${CUDA_TAG}" "${GPU_TUNED_VARIANT}" || return 1 2>/dev/null || exit 1
GPU_TUNED_BUILD_ROOT="${GPU_TUNED_REPO_ROOT}/tuned/build/${CUDA_TAG}"
GPU_TUNED_VENV_DIR="${GPU_TUNED_BUILD_ROOT}/venv"
GPU_TUNED_WHEEL_DIR="${GPU_TUNED_BUILD_ROOT}/wheel"
GPU_TUNED_CMAKE_DIR="$(gpu_tuned_out_dir build "${GPU_TUNED_REPO_ROOT}" "${CUDA_TAG}")"
GPU_TUNED_DIST_DIR="$(gpu_tuned_out_dir dist "${GPU_TUNED_REPO_ROOT}" "${CUDA_TAG}")"
GPU_TUNED_RELEASES_DIR="$(gpu_tuned_out_dir releases "${GPU_TUNED_REPO_ROOT}" "${CUDA_TAG}")"
export GPU_TUNED_REPO_ROOT GPU_TUNED_BUILD_ROOT GPU_TUNED_VENV_DIR GPU_TUNED_WHEEL_DIR GPU_TUNED_CMAKE_DIR
export GPU_TUNED_DIST_DIR GPU_TUNED_RELEASES_DIR

# ccache is per toolkit (C4). sglang-kernel's CMakeLists enables ccache only when CCACHE_DIR is set.
export CCACHE_DIR="${GPU_TUNED_BUILD_ROOT}/ccache"
export CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-60G}"

# sglang's runtime JIT kernels (python/sglang/kernels/jit) look up prebuilt modules here. The build seeds it;
# at run time SGLANG_CRASH_ON_JIT_COMPILE=1 turns any miss into an error instead of an nvcc compile.
export SGLANG_JIT_CACHE_DIR="${SGLANG_JIT_CACHE_DIR:-${GPU_TUNED_BUILD_ROOT}/jit-cache}"

# --- Build parallelism: gpu_tuned_jobs unless MAX_JOBS is set (C5). One nvcc thread per job keeps MAX_JOBS the
#     real number of compiles (sglang-kernel's default is --threads=32 per nvcc, which oversubscribes). ---
export MAX_JOBS="${MAX_JOBS:-$(gpu_tuned_jobs)}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-${MAX_JOBS}}"

echo "[tuned/env] GPU_TUNED_VARIANT=${GPU_TUNED_VARIANT} arch=${GPU_TUNED_CUDA_ARCH} CUDA_HOME=${CUDA_HOME} MAX_JOBS=${MAX_JOBS} tree=${GPU_TUNED_BUILD_ROOT}"
