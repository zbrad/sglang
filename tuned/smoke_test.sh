#!/bin/bash
# tuned/smoke_test.sh <variant> — check the PACKAGED wheels (tuned/build/<cuda-tag>/dist/) as a consumer receives them,
# after tuned/package.sh and before tuned/release.sh, and write tuned/build/<cuda-tag>/releases/SMOKE_RESULTS_<variant>.log.
#
#   CUDA_TAG=cu134 bash tuned/smoke_test.sh gb10
#
# Checks: both sidecars verify; the packaged wheels install --no-deps into the build venv; sgl_kernel and sglang load
# from site-packages; every sgl_kernel device-code .so carries the build-info stamp for this release and fork commit;
# a fresh interpreter gets the strict defaults from sglang_tuned.pth (cache dir inside the installed package); the
# packaged METADATA carries the tuned torch / flashinfer / sglang-kernel requirements and none of the dropped ones; and
# the acceptance stand-ins run in strict mode from the SHIPPED cache (SGLANG_JIT_CACHE_DIR unset). The unstamped wheels
# are reinstalled at the end, so the build venv is left as full_test.sh expects.
set -uo pipefail

GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/smoke_test.sh <variant>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

# shellcheck source=env.sh
source "${REPO_ROOT}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}" || exit 1

INFO="${GPU_TUNED_DIST_DIR}/package-info.txt"
[[ -f "${INFO}" ]] || { echo "ERROR: ${INFO} not found; run CUDA_TAG=${CUDA_TAG} bash tuned/package.sh ${GPU_TUNED_VARIANT} first." >&2; exit 1; }
info() { sed -n "s/^$1=//p" "${INFO}"; }
KERNEL_WHEEL="${GPU_TUNED_DIST_DIR}/$(info wheel)"
SGLANG_WHEEL="${GPU_TUNED_DIST_DIR}/$(info wheel_sglang)"
VERSION="$(info version)"
PY="${GPU_TUNED_VENV_DIR}/bin/python"
mkdir -p "${GPU_TUNED_RELEASES_DIR}"
LOG="${GPU_TUNED_RELEASES_DIR}/SMOKE_RESULTS_${GPU_TUNED_VARIANT}.log"
rm -f "${LOG}"
export CUDA_VISIBLE_DEVICES="${GPU_TUNED_TEST_GPUS:-0}"

FAILS=0
ok()   { echo "  ok:   $*"; }
fail() { echo "  FAIL: $*"; FAILS=$((FAILS + 1)); }

smoke() {
    echo "sglang tuned smoke test -- $(info release_tag) variant=${GPU_TUNED_VARIANT} ${CUDA_TAG}"
    echo "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "commit:  $(git rev-parse HEAD) (packaged at $(info commit))"
    local w so site pth_out meta
    for w in "${KERNEL_WHEEL}" "${SGLANG_WHEEL}"; do
        echo "wheel:   $(basename "${w}") sha256 $(sha256sum "${w}" | awk '{print $1}')"
        gpu_tuned_verify_sha256_sidecar "${w}" && ok "$(basename "${w}").sha256 verifies" || fail "$(basename "${w}").sha256 does not verify"
    done
    "${PY}" -m pip install --quiet --no-deps --force-reinstall "${KERNEL_WHEEL}" "${SGLANG_WHEEL}" \
        && ok "packaged wheels installed (--no-deps)" || { fail "packaged wheels did not install"; return; }
    site="$(cd "${GPU_TUNED_BUILD_ROOT}" && "${PY}" -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

    while IFS= read -r so; do
        gpu_tuned_so_has_device_code "${so}" || continue
        if gpu_tuned_verify_build_info_json "${so}" "release=${VERSION}" "variant=${GPU_TUNED_VARIANT}" "cuda.tag=${CUDA_TAG}" "fork.commit=$(info built_commit)" >/dev/null; then
            ok "stamp: ${so#"${site}"/}"
        else
            fail "stamp missing or wrong: ${so#"${site}"/}"
        fi
    done < <(find "${site}/sgl_kernel" -name '*.so')

    # A fresh interpreter, no SGLANG_* or FLASHINFER_* in its environment: the .pth must supply the strict defaults.
    pth_out="$(cd "${GPU_TUNED_BUILD_ROOT}" && env -u SGLANG_JIT_CACHE_DIR -u SGLANG_CRASH_ON_JIT_COMPILE -u FLASHINFER_DISABLE_JIT -u SGLANG_ENABLE_JIT_DEEPGEMM \
        "${PY}" -c 'import os; print(os.environ.get("SGLANG_JIT_CACHE_DIR")); print(os.environ.get("SGLANG_CRASH_ON_JIT_COMPILE"), os.environ.get("FLASHINFER_DISABLE_JIT"), os.environ.get("SGLANG_ENABLE_JIT_DEEPGEMM"))')"
    if [[ "$(sed -n 1p <<< "${pth_out}")" == "${site}/sglang/kernels/jit/_tuned_cache" && "$(sed -n 2p <<< "${pth_out}")" == "1 1 0" ]]; then
        ok "sglang_tuned.pth defaults: cache in the installed package; crash-on-JIT 1, flashinfer JIT off, DeepGEMM off"
    else
        fail "sglang_tuned.pth defaults wrong: ${pth_out//$'\n'/ | }"
    fi

    meta="$(ls "${site}"/sglang-*.dist-info/METADATA 2>/dev/null | head -1)"
    if [[ -n "${meta}" ]] && grep -q '^Requires-Dist: torch @ https://github.com/zbrad/pytorch/' "${meta}" \
        && grep -q '^Requires-Dist: sglang-kernel @ https://github.com/zbrad/sglang/' "${meta}" \
        && grep -q '^Requires-Dist: torchvision @ https://github.com/zbrad/vision/' "${meta}" \
        && ! grep -qiE '^Requires-Dist: (sgl-flash-mla|sgl-deep-gemm|sgl-deep-ep|torchaudio|torch==|pillow<)' "${meta}" \
        && ! grep -qiE '^Requires-Dist: torchvision( *[=<>!~;]|$)' "${meta}"; then
        ok "sglang METADATA: tuned torch, torchvision and sglang-kernel by URL, dropped dependencies absent"
    else
        fail "sglang METADATA does not carry the tuned dependency set (${meta:-no METADATA})"
    fi

    echo "=== acceptance stand-ins, strict, from the shipped cache ==="
    if (cd "${GPU_TUNED_BUILD_ROOT}" && env -u SGLANG_JIT_CACHE_DIR SGLANG_JIT_CACHE_DIR="${site}/sglang/kernels/jit/_tuned_cache" \
            "${PY}" -I "${REPO_ROOT}/tuned/gb10_acceptance.py" strict); then
        ok "acceptance stand-ins pass in strict mode from the shipped cache"
    else
        fail "acceptance stand-ins failed in strict mode from the shipped cache"
    fi

    # Leave the venv as full_test.sh expects: the unstamped wheels.
    "${PY}" -m pip install --quiet --no-deps --force-reinstall "${GPU_TUNED_WHEEL_DIR}"/sglang_kernel-*.whl "${GPU_TUNED_WHEEL_DIR}"/sglang-*.whl \
        || fail "could not reinstall the unstamped wheels"

    echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if [[ "${FAILS}" -eq 0 ]]; then echo "=== Smoke test summary: passed ==="; else echo "=== Smoke test summary: FAILED (${FAILS}) ==="; fi
}

smoke 2>&1 | tee "${LOG}"
echo "Results written to ${LOG}"
grep -q '^=== Smoke test summary: passed ===$' "${LOG}"
