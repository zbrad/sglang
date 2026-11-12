#!/bin/bash
# tuned/full_test.sh <variant> — test the wheels tuned/build.sh made, for ONE variant and ONE CUDA toolkit, and write
# a results log in that toolkit's tree (tuned/build/<cuda-tag>/releases/TEST_RESULTS_<variant>.log).
#
#   CUDA_TAG=cu134 bash tuned/full_test.sh gb10
#
# The unstamped wheels (tuned/build/<cuda-tag>/wheel/) are installed --no-deps into the toolkit's build venv, and every
# test runs from the toolkit tree, never the source tree, so sglang and sgl_kernel are the wheels'. The venv holds the
# dependency set build.sh installed (tuned/requirements-<variant>.txt plus upstream's remaining dependencies).
#
# Scope (narrower than upstream's test/ suite, which is not run; same shape as zbrad/vllm's recorded G3 decision):
#   1. gates: wheel labels and build commit, H6 source check, sgl_kernel loads from the wheel, its device-code shared
#      objects are sm_<arch> only with no PTX, toolkit libraries resolve under CUDA_HOME;
#   2. tuned/gb10_acceptance.py strict: the stand-in models load and generate with SGLANG_CRASH_ON_JIT_COMPILE=1,
#      FLASHINFER_DISABLE_JIT=1 and DeepGEMM off, from the JIT cache build.sh seeded (needs a Hugging Face download);
#   3. python -m pytest tuned/tests: the tuned tooling's own tests (no GPU).
# Skips by design, named in the log header: upstream's test/ suite, multi-GPU behaviour (single-GPU build), and Triton's
# own runtime compiles (Triton is a JIT compiler by design; out of the no-JIT scope, as in zbrad/vllm).
#
# The log ends with "=== Full test suite summary: acceptance + tuned tests passed ===" only when every step passed;
# release.sh greps for that exact line. The other outcome is "=== Full test suite summary: FAILED (see above) ===".
set -uo pipefail

GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/full_test.sh <variant>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}" || exit 1

# shellcheck source=env.sh
source "${REPO_ROOT}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}" || exit 1

KERNEL_WHEEL="$(ls "${GPU_TUNED_WHEEL_DIR}"/sglang_kernel-*.whl 2>/dev/null | head -1)"
SGLANG_WHEEL="$(ls "${GPU_TUNED_WHEEL_DIR}"/sglang-*.whl 2>/dev/null | head -1)"
[[ -n "${KERNEL_WHEEL}" && -n "${SGLANG_WHEEL}" ]] || { echo "ERROR: wheels missing in ${GPU_TUNED_WHEEL_DIR}. Run CUDA_TAG=${CUDA_TAG} bash tuned/build.sh ${GPU_TUNED_VARIANT} first." >&2; exit 1; }
BUILD_COMMIT="$(cat "${GPU_TUNED_BUILD_ROOT}/build-commit" 2>/dev/null || true)"
LABEL="$(cat "${GPU_TUNED_BUILD_ROOT}/release-label" 2>/dev/null || true)"
[[ -n "${BUILD_COMMIT}" && -n "${LABEL}" ]] || { echo "ERROR: ${GPU_TUNED_BUILD_ROOT}/build-commit or release-label missing: the last build did not finish. Re-run tuned/build.sh ${GPU_TUNED_VARIANT}." >&2; exit 1; }
[[ "$(cat "${GPU_TUNED_BUILD_ROOT}/jit-cache-seeded" 2>/dev/null || true)" == "${LABEL}" ]] \
    || { echo "ERROR: the JIT cache was not seeded for label ${LABEL} (${GPU_TUNED_BUILD_ROOT}/jit-cache-seeded). Re-run tuned/build.sh ${GPU_TUNED_VARIANT}." >&2; exit 1; }
for w in "${KERNEL_WHEEL}" "${SGLANG_WHEEL}"; do
    [[ "$(basename "${w}")" == *"+${GPU_TUNED_VARIANT}.cu${CUDA_VERSION_COMPACT}.${LABEL}-"* ]] \
        || { echo "ERROR: $(basename "${w}") does not carry label ${GPU_TUNED_VARIANT}.cu${CUDA_VERSION_COMPACT}.${LABEL} from release-label." >&2; exit 1; }
done
[[ "${BUILD_COMMIT}" != *-dirty ]] || { echo "ERROR: the build was made from a dirty tree (${BUILD_COMMIT}); commit and rebuild." >&2; exit 1; }
git cat-file -e "${BUILD_COMMIT}^{commit}" 2>/dev/null || { echo "ERROR: build commit ${BUILD_COMMIT} is not in this checkout." >&2; exit 1; }
# H6: the wheels must be from the source at HEAD; only tuned/ may differ.
STALE_SRC="$(git diff --name-only "${BUILD_COMMIT}" HEAD -- . ':!tuned' | head -5)"
if [[ -n "${STALE_SRC}" ]]; then
    echo "ERROR: the wheels are from ${BUILD_COMMIT}, but source outside tuned/ changed since:" >&2
    echo "${STALE_SRC}" | sed 's/^/    /' >&2
    echo "  Rebuild (tuned/build.sh ${GPU_TUNED_VARIANT}) before testing." >&2
    exit 1
fi

PY="${GPU_TUNED_VENV_DIR}/bin/python"
[[ -x "${PY}" ]] || { echo "ERROR: ${PY} not found. Run CUDA_TAG=${CUDA_TAG} bash tuned/build.sh ${GPU_TUNED_VARIANT} first." >&2; exit 1; }
mkdir -p "${GPU_TUNED_RELEASES_DIR}"
RESULTS_FILE="${GPU_TUNED_RELEASES_DIR}/TEST_RESULTS_${GPU_TUNED_VARIANT}.log"
rm -f "${RESULTS_FILE}"

# Tuned builds are single-GPU: the tests see exactly one device (GPU_TUNED_TEST_GPUS, default 0).
export CUDA_VISIBLE_DEVICES="${GPU_TUNED_TEST_GPUS:-0}"

echo "installing pytest and the two wheels into the build venv (--no-deps)"
"${PY}" -m pip install --quiet pytest || { echo "ERROR: pip could not install pytest." >&2; exit 1; }
"${PY}" -m pip install --quiet --no-deps --force-reinstall "${KERNEL_WHEEL}" "${SGLANG_WHEEL}" \
    || { echo "ERROR: pip could not install the wheels." >&2; exit 1; }
SITE_DIR="$(cd "${GPU_TUNED_BUILD_ROOT}" && "${PY}" -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

{
    echo "sglang tuned test run -- variant=${GPU_TUNED_VARIANT} cuda=${CUDA_VER} (${CUDA_TAG})"
    echo "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "commit:  $(git rev-parse HEAD)"
    echo "build:   ${BUILD_COMMIT} (label ${LABEL})"
    for w in "${KERNEL_WHEEL}" "${SGLANG_WHEEL}"; do echo "wheel:   $(basename "${w}") sha256 $(sha256sum "${w}" | awk '{print $1}')"; done
    echo "torch:   $(cd "${GPU_TUNED_BUILD_ROOT}" && "${PY}" -I -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no device")' 2>&1 | tail -1)"
    echo "flashinfer: $("${PY}" -m pip freeze 2>/dev/null | grep -iE '^flashinfer' | tr '\n' ' ')"
    echo "jit cache: ${SGLANG_JIT_CACHE_DIR} ($(find "${SGLANG_JIT_CACHE_DIR}" -name '*.so' | wc -l) modules, seeded for ${LABEL})"
    echo "scope:   gates; tuned/gb10_acceptance.py strict (stand-ins); python -m pytest tuned/tests"
    echo "devices: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (single-GPU build: multi-GPU tests are not part of its scope)"
    echo "skips by design: upstream's test/ suite; multi-GPU; Triton's own runtime compiles (out of the no-JIT scope)"
    echo ""
} > "${RESULTS_FILE}"

# Gates. A function, so its status comes back through the tee pipeline's PIPESTATUS.
gate_checks() {
    local rc=0 loaded_from so n=0 ptx
    loaded_from="$(cd "${GPU_TUNED_BUILD_ROOT}" && "${PY}" -I -c 'import sgl_kernel, sglang; print(sgl_kernel.__file__); print(sglang.__file__)' 2>&1)" || { echo "  FAIL: import sgl_kernel/sglang: ${loaded_from}"; return 1; }
    while IFS= read -r so; do
        if [[ "${so}" == "${SITE_DIR}/"* ]]; then echo "  ok:   loads from the wheel: ${so}"
        else echo "  FAIL: '${so}' is not under ${SITE_DIR}"; rc=1; fi
    done <<< "${loaded_from}"
    while IFS= read -r so; do
        gpu_tuned_so_has_device_code "${so}" || continue
        n=$((n + 1))
        gpu_tuned_verify_arch "${so}" "${GPU_TUNED_CUDA_ARCH}" || rc=1
        ptx="$(cuobjdump --list-ptx "${so}" 2>&1 || true)"
        if [[ -n "$(printf '%s\n' "${ptx}" | grep -E '^PTX file +[0-9]+:' || true)" ]]; then echo "  FAIL: ${so#"${SITE_DIR}"/} embeds PTX"; rc=1; fi
        gpu_tuned_verify_toolkit_libs "${so}" || rc=1
    done < <(find "${SITE_DIR}/sgl_kernel" -name '*.so')
    if [[ "${n}" -gt 0 ]]; then echo "  ok:   ${n} sgl_kernel shared objects checked (sm_${GPU_TUNED_CUDA_ARCH} only, no PTX)"
    else echo "  FAIL: no sgl_kernel shared object with device code"; rc=1; fi
    return "${rc}"
}
logrun() { "$@" 2>&1 | tee -a "${RESULTS_FILE}"; return "${PIPESTATUS[0]}"; }

STATUS=0
echo "=== gates ===" | tee -a "${RESULTS_FILE}"
logrun gate_checks || STATUS=1
echo "" | tee -a "${RESULTS_FILE}"
echo "=== tuned/gb10_acceptance.py strict (stand-ins; needs a Hugging Face download) ===" | tee -a "${RESULTS_FILE}"
(cd "${GPU_TUNED_BUILD_ROOT}" && "${PY}" -I "${REPO_ROOT}/tuned/gb10_acceptance.py" strict) 2>&1 | tee -a "${RESULTS_FILE}"
[[ ${PIPESTATUS[0]} -eq 0 ]] || { STATUS=1; echo "gb10_acceptance.py strict FAILED" | tee -a "${RESULTS_FILE}"; }

echo "" | tee -a "${RESULTS_FILE}"
echo "=== python -m pytest tuned/tests ===" | tee -a "${RESULTS_FILE}"
(cd "${REPO_ROOT}" && "${PY}" -m pytest -q -p no:cacheprovider tuned/tests) 2>&1 | tee -a "${RESULTS_FILE}"
[[ ${PIPESTATUS[0]} -eq 0 ]] || { STATUS=1; echo "tuned/tests FAILED" | tee -a "${RESULTS_FILE}"; }

echo "" >> "${RESULTS_FILE}"
echo "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${RESULTS_FILE}"
if [[ "${STATUS}" -eq 0 ]]; then
    echo "=== Full test suite summary: acceptance + tuned tests passed ===" | tee -a "${RESULTS_FILE}"
else
    echo "=== Full test suite summary: FAILED (see above) ===" | tee -a "${RESULTS_FILE}"
fi
echo "Results written to ${RESULTS_FILE}"
exit "${STATUS}"
