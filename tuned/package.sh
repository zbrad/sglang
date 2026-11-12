#!/bin/bash
# tuned/package.sh <variant> — package the two wheels tuned/build.sh made and tuned/full_test.sh tested: stamp STAGED
# copies, rewrite their dependency metadata for the variant, add the seeded JIT cache and its runtime defaults to the
# sglang wheel, repack, write sha256 sidecars, package-info.txt and, if no notes are tracked yet, a DRAFT release-notes
# file. It never builds, tests, tags or publishes (B3). Stage order:
#   build.sh -> full_test.sh -> package.sh -> smoke_test.sh -> (review and commit the notes) -> release.sh -> publish.sh
#
#   CUDA_TAG=cu134 bash tuned/package.sh gb10
#
# sglang-kernel wheel: every shared object with device code is checked (sm_<arch> only, no PTX) and stamped with the
#   JSON build info (H7); Requires-Dist torch==<upstream pin> becomes the tuned torch line of tuned/requirements-<variant>.txt.
# sglang wheel: Requires-Dist entries named in GPU_TUNED_DROP_DEPS are replaced by the requirements file's lines plus
#   sglang-kernel @ <this release's kernel wheel URL> (tuned/deps.py); the JIT cache build.sh seeded is added under
#   sglang/kernels/jit/_tuned_cache/; and sglang_tuned.pth sets, only where the environment does not already, the
#   strict runtime defaults (SGLANG_JIT_CACHE_DIR at that cache, SGLANG_CRASH_ON_JIT_COMPILE=1, FLASHINFER_DISABLE_JIT=1,
#   SGLANG_ENABLE_JIT_DEEPGEMM=0). No upstream source file is changed for any of this.
# Output: tuned/build/<cuda-tag>/dist/{both wheels, their .sha256, package-info.txt}; package-info.txt is written last
#   (H2): its presence means every check above it passed.
set -euo pipefail

GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/package.sh <variant>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=env.sh
source "${REPO_ROOT}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}"

[[ -d "${GPU_TUNED_VENV_DIR}" ]] || { echo "ERROR: ${GPU_TUNED_VENV_DIR} not found. Run tuned/build.sh ${GPU_TUNED_VARIANT} first." >&2; exit 1; }
gpu_tuned_verify_venv "${GPU_TUNED_VENV_DIR}" "${REPO_ROOT}"
gpu_tuned_check_venv_setuptools "${GPU_TUNED_VENV_DIR}" || exit 1
# shellcheck source=/dev/null
source "${GPU_TUNED_VENV_DIR}/bin/activate"

KERNEL_BUILT="$(ls "${GPU_TUNED_WHEEL_DIR}"/sglang_kernel-*.whl 2>/dev/null | head -1)"
SGLANG_BUILT="$(ls "${GPU_TUNED_WHEEL_DIR}"/sglang-*.whl 2>/dev/null | head -1)"
[[ -n "${KERNEL_BUILT}" && -n "${SGLANG_BUILT}" ]] || { echo "ERROR: wheels missing in ${GPU_TUNED_WHEEL_DIR}. Run CUDA_TAG=${CUDA_TAG} bash tuned/build.sh ${GPU_TUNED_VARIANT} first." >&2; exit 1; }
LABEL="$(cat "${GPU_TUNED_BUILD_ROOT}/release-label" 2>/dev/null || true)"
BUILD_COMMIT="$(cat "${GPU_TUNED_BUILD_ROOT}/build-commit" 2>/dev/null || true)"
[[ -n "${LABEL}" && -n "${BUILD_COMMIT}" ]] || { echo "ERROR: release-label or build-commit missing in ${GPU_TUNED_BUILD_ROOT}: the last build did not finish." >&2; exit 1; }
[[ "${BUILD_COMMIT}" != *-dirty ]] || { echo "ERROR: the build was made from a dirty tree (${BUILD_COMMIT}); commit and rebuild." >&2; exit 1; }
[[ "$(cat "${GPU_TUNED_BUILD_ROOT}/jit-cache-seeded" 2>/dev/null || true)" == "${LABEL}" ]] \
    || { echo "ERROR: the JIT cache was not seeded for ${LABEL}; re-run tuned/build.sh ${GPU_TUNED_VARIANT}." >&2; exit 1; }

# The label is the one build.sh allocated and baked in; it is never recomputed here (E4).
KERNEL_NAME="$(basename "${KERNEL_BUILT}")"; SGLANG_NAME="$(basename "${SGLANG_BUILT}")"
KERNEL_VERSION="$(gpu_tuned_wheel_version "${KERNEL_BUILT}" sglang_kernel)"
VERSION="$(gpu_tuned_wheel_version "${SGLANG_BUILT}" sglang)"
for v in "${KERNEL_VERSION}" "${VERSION}"; do
    [[ "${v}" == *"+${GPU_TUNED_VARIANT}.cu${CUDA_VERSION_COMPACT}.${LABEL}" ]] \
        || { echo "ERROR: version ${v} does not carry +${GPU_TUNED_VARIANT}.cu${CUDA_VERSION_COMPACT}.${LABEL} from release-label." >&2; exit 1; }
done
gpu_tuned_check_wheel_label "${KERNEL_BUILT}"
gpu_tuned_check_wheel_label "${SGLANG_BUILT}"
BASE_VERSION="${VERSION%%+*}"
RELEASE_TAG="v${VERSION}"
gpu_tuned_check_release_tag "${RELEASE_TAG}"
RELEASE_URL="https://github.com/zbrad/sglang/releases/download/$(printf '%s' "${RELEASE_TAG}" | sed 's/+/%2B/g')"

# H6: refuse a build from other source than HEAD; HEAD may differ from the build only in tuned/.
git cat-file -e "${BUILD_COMMIT}^{commit}" 2>/dev/null || { echo "ERROR: build commit ${BUILD_COMMIT} is not in this checkout." >&2; exit 1; }
STALE_SRC="$(git diff --name-only "${BUILD_COMMIT}" HEAD -- . ':!tuned' | head -5)"
[[ -z "${STALE_SRC}" ]] || { echo "ERROR: the wheels are from ${BUILD_COMMIT}, but source outside tuned/ changed since:" >&2; echo "${STALE_SRC}" | sed 's/^/    /' >&2; exit 1; }
TEST_LOG="${GPU_TUNED_RELEASES_DIR}/TEST_RESULTS_${GPU_TUNED_VARIANT}.log"
[[ -f "${TEST_LOG}" ]] || { echo "ERROR: ${TEST_LOG} not found; run CUDA_TAG=${CUDA_TAG} bash tuned/full_test.sh ${GPU_TUNED_VARIANT} before packaging." >&2; exit 1; }
grep -q "^=== Full test suite summary: acceptance + tuned tests passed ===\$" "${TEST_LOG}" \
    || { echo "ERROR: ${TEST_LOG} does not show a full, passing run." >&2; exit 1; }
for w in "${KERNEL_BUILT}" "${SGLANG_BUILT}"; do
    [[ "${w}" -nt "${TEST_LOG}" ]] && { echo "ERROR: $(basename "${w}") is newer than the test log; re-run tuned/full_test.sh ${GPU_TUNED_VARIANT}." >&2; exit 1; }
    grep -q "^wheel:   $(basename "${w}") sha256 $(sha256sum "${w}" | awk '{print $1}')\$" "${TEST_LOG}" \
        || { echo "ERROR: the test log did not test this $(basename "${w}") (sha256 differs or missing)." >&2; exit 1; }
done
TESTED_COMMIT="$(sed -n 's/^commit: *//p' "${TEST_LOG}" | head -1)"

echo "=========================================="
echo "Packaging sglang wheels (${GPU_TUNED_HW_LABEL})"
echo "=========================================="
echo "Versions:    sglang ${VERSION}, sglang-kernel ${KERNEL_VERSION}"
echo "Release tag: ${RELEASE_TAG}"
echo "Built at:    ${BUILD_COMMIT}; tested at ${TESTED_COMMIT}; packaging at $(git rev-parse HEAD)"
echo ""

mkdir -p "${GPU_TUNED_DIST_DIR}"
rm -f "${GPU_TUNED_DIST_DIR}/package-info.txt" "${GPU_TUNED_DIST_DIR}"/sglang_kernel-*.whl* "${GPU_TUNED_DIST_DIR}"/sglang-*.whl*
command -v readelf >/dev/null && command -v objcopy >/dev/null && command -v cuobjdump >/dev/null \
    || { echo "ERROR: readelf, objcopy and cuobjdump are needed for the packaging checks (H4)." >&2; exit 1; }

STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT
TORCH_REQ="${STAGE}/torch-req.txt"
grep -E '^torch @ ' "${GPU_TUNED_REQUIREMENTS}" > "${TORCH_REQ}" || { echo "ERROR: no 'torch @ <url>' line in ${GPU_TUNED_REQUIREMENTS}." >&2; exit 1; }

# --- sglang-kernel ---
python3 -m wheel unpack "${KERNEL_BUILT}" --dest "${STAGE}/kernel"
KUNPACKED="$(find "${STAGE}/kernel" -maxdepth 1 -mindepth 1 -type d)"
STOCK_RE='^lib(c10|c10_cuda|torch|torch_cpu|torch_cuda|torch_cuda_linalg|torch_python|torch_global_deps|caffe2_nvrtc|shm)\.so(\.[0-9]+)*$'
leaks="$(find "${KUNPACKED}" -type f -name '*.so*' -exec basename {} \; | grep -E "${STOCK_RE}" || true)"
[[ -z "${leaks}" ]] || { echo "ERROR: sglang-kernel ships shared objects under stock torch names: ${leaks}" >&2; exit 1; }
STAMPED=0
while IFS= read -r so; do
    gpu_tuned_so_has_device_code "${so}" || continue
    gpu_tuned_verify_arch "${so}" "${GPU_TUNED_CUDA_ARCH}" || exit 1
    PTX_OUT="$(cuobjdump --list-ptx "${so}" 2>&1)" \
        || { echo "ERROR: cuobjdump --list-ptx failed on ${so#"${STAGE}"/}: ${PTX_OUT}" >&2; exit 1; }
    [[ -z "$(printf '%s\n' "${PTX_OUT}" | grep -E '^PTX file +[0-9]+:' || true)" ]] \
        || { echo "ERROR: ${so#"${STAGE}"/} embeds PTX: the driver would JIT-compile it at run time." >&2; exit 1; }
    gpu_tuned_stamp_lib_json "${so}" sglang-kernel "${VERSION}" "" \
        "upstream.declared_version=${KERNEL_VERSION%%+*}" "fork.commit=${BUILD_COMMIT}" "fork.dirty:bool=false" \
        || { echo "ERROR: could not stamp ${so#"${STAGE}"/}." >&2; exit 1; }
    STAMPED=$((STAMPED + 1))
done < <(find "${KUNPACKED}" -name '*.so')
[[ "${STAMPED}" -gt 0 ]] || { echo "ERROR: no device-code shared object found in ${KERNEL_NAME}." >&2; exit 1; }
echo "OK: ${STAMPED} sglang-kernel shared objects checked and stamped"
python3 -I tuned/deps.py rewrite "$(ls -d "${KUNPACKED}"/*.dist-info)/METADATA" "${TORCH_REQ}" torch
python3 -m wheel pack "${KUNPACKED}" --dest-dir "${GPU_TUNED_DIST_DIR}"

# --- sglang ---
python3 -m wheel unpack "${SGLANG_BUILT}" --dest "${STAGE}/sglang"
SUNPACKED="$(find "${STAGE}/sglang" -maxdepth 1 -mindepth 1 -type d)"
SGLANG_REQ="${STAGE}/sglang-req.txt"
{ cat "${GPU_TUNED_REQUIREMENTS}"; echo "sglang-kernel @ ${RELEASE_URL}/${KERNEL_NAME}"; } > "${SGLANG_REQ}"
# shellcheck disable=SC2086 # GPU_TUNED_DROP_DEPS is a word list
python3 -I tuned/deps.py rewrite "$(ls -d "${SUNPACKED}"/*.dist-info)/METADATA" "${SGLANG_REQ}" ${GPU_TUNED_DROP_DEPS}
CACHE_DEST="${SUNPACKED}/sglang/kernels/jit/_tuned_cache"
[[ ! -e "${CACHE_DEST}" ]] || { echo "ERROR: ${SGLANG_NAME} already has sglang/kernels/jit/_tuned_cache." >&2; exit 1; }
mkdir -p "${CACHE_DEST}"
cp -a "${SGLANG_JIT_CACHE_DIR}/." "${CACHE_DEST}/"
find "${CACHE_DEST}" \( -name '.lock' -o -name '.staging-*' \) -prune -exec rm -rf {} +
MODULES="$(find "${CACHE_DEST}" -name '*.so' | wc -l)"
[[ "${MODULES}" -gt 0 ]] || { echo "ERROR: the seeded JIT cache ${SGLANG_JIT_CACHE_DIR} has no modules." >&2; exit 1; }
# One line, as .pth requires: environment defaults at interpreter start, never overriding what is already set.
cat > "${SUNPACKED}/sglang_tuned.pth" <<'PTH'
import os, importlib.util as _u; _s = _u.find_spec("sglang"); _s and _s.submodule_search_locations and [os.environ.setdefault(_k, _v) for _k, _v in (("SGLANG_JIT_CACHE_DIR", os.path.join(list(_s.submodule_search_locations)[0], "kernels", "jit", "_tuned_cache")), ("SGLANG_CRASH_ON_JIT_COMPILE", "1"), ("FLASHINFER_DISABLE_JIT", "1"), ("SGLANG_ENABLE_JIT_DEEPGEMM", "0"))]
PTH
echo "OK: ${MODULES} JIT modules and sglang_tuned.pth added to the sglang wheel"
python3 -m wheel pack "${SUNPACKED}" --dest-dir "${GPU_TUNED_DIST_DIR}"

for name in "${KERNEL_NAME}" "${SGLANG_NAME}"; do
    [[ -f "${GPU_TUNED_DIST_DIR}/${name}" ]] || { echo "ERROR: repacked wheel is not ${name}." >&2; exit 1; }
    gpu_tuned_check_wheel_label "${GPU_TUNED_DIST_DIR}/${name}"
    gpu_tuned_write_sha256_sidecar "${GPU_TUNED_DIST_DIR}/${name}" >/dev/null
done
KERNEL_SHA="$(awk '{print $1}' "${GPU_TUNED_DIST_DIR}/${KERNEL_NAME}.sha256")"
SGLANG_SHA="$(awk '{print $1}' "${GPU_TUNED_DIST_DIR}/${SGLANG_NAME}.sha256")"

# Dependency audit of the packaged sglang wheel's resolved runtime deps (H9); the kernel wheel is found beside it.
gpu_tuned_audit_deps "${GPU_TUNED_DIST_DIR}/${SGLANG_NAME}" "${GPU_TUNED_BUILD_ROOT}/dependency-audit.log" --find-links "${GPU_TUNED_DIST_DIR}" \
    || { echo "ERROR: dependency audit failed (see ${GPU_TUNED_BUILD_ROOT}/dependency-audit.log)." >&2; exit 1; }

# H5: a DRAFT notes file if none is tracked yet; release.sh refuses it until a person has reviewed it.
NOTES="tuned/releases/RELEASE_NOTES_${RELEASE_TAG#v}.md"
if [[ ! -f "${NOTES}" ]]; then
    mkdir -p tuned/releases
    {
        echo "<!-- DRAFT — not yet published: review every line, then delete this banner and commit. -->"
        echo "# sglang ${BASE_VERSION} — ${GPU_TUNED_VARIANT} ${CUDA_TAG} (${RELEASE_TAG#v})"
        echo ""
        echo "Single-GPU sglang for ${GPU_TUNED_HW_LABEL}, CUDA ${CUDA_VER}: sglang-kernel built for sm_${GPU_TUNED_CUDA_ARCH} only"
        echo "(SASS, no PTX), and no nvcc at run time (sglang's JIT kernels come from a cache seeded at build time)."
        echo ""
        echo "- Release label: ${LABEL} (UTC date of the tuned commit)"
        echo "- Built from commit ${BUILD_COMMIT}; tested at ${TESTED_COMMIT}"
        echo "- Upstream base: ${GPU_TUNED_UPSTREAM_REF} merge-base $(git merge-base HEAD "${GPU_TUNED_UPSTREAM_REF}" 2>/dev/null || echo unknown)"
        echo "- Dependencies: ${GPU_TUNED_REQUIREMENTS}; dropped from upstream's list: ${GPU_TUNED_DROP_DEPS} (reasons in tuned/repo.env)"
        echo "- Seeded JIT modules: ${MODULES}"
        echo "- sha256: ${KERNEL_NAME} ${KERNEL_SHA}; ${SGLANG_NAME} ${SGLANG_SHA}"
        echo ""
        echo "## Runtime requirements"
        echo ""
        echo "sglang's JIT cache key fingerprints \`nvcc --version\` and \`c++ --version\`: the host needs the CUDA ${CUDA_VER} toolkit"
        echo "(nvcc is asked its version, never run as a compiler) and the same host compiler as the build, or every lookup misses"
        echo "and, with SGLANG_CRASH_ON_JIT_COMPILE=1, fails."
        echo ""
        echo "## Tests"
        echo ""
        echo "Scope (from the attached test log): $(sed -n 's/^scope: *//p' "${TEST_LOG}" | head -1)"
        echo "Skips by design: $(sed -n 's/^skips by design: *//p' "${TEST_LOG}" | head -1)"
        echo "Result: $(grep '^=== Full test suite summary' "${TEST_LOG}" | tail -1)"
        echo ""
        echo "## Install"
        echo ""
        echo '```'
        echo "pip install \"sglang @ ${RELEASE_URL}/${SGLANG_NAME}\""
        echo '```'
    } > "${NOTES}"
    echo "Wrote DRAFT release notes: ${NOTES}"
fi

# Written last.
cat > "${GPU_TUNED_DIST_DIR}/package-info.txt" <<INFO
release_tag=${RELEASE_TAG}
wheel=${KERNEL_NAME}
wheel_sglang=${SGLANG_NAME}
version=${VERSION}
kernel_version=${KERNEL_VERSION}
upstream_version=${BASE_VERSION}
label=${LABEL}
commit=$(git rev-parse HEAD)
built_commit=${BUILD_COMMIT}
tested_commit=${TESTED_COMMIT}
cuda_tag=${CUDA_TAG}
variant=${GPU_TUNED_VARIANT}
sha256=${KERNEL_SHA}
sha256_sglang=${SGLANG_SHA}
jit_modules=${MODULES}
INFO

echo ""
echo "Packaged (not published): ${GPU_TUNED_DIST_DIR}/{${KERNEL_NAME},${SGLANG_NAME}} (${RELEASE_TAG})"
echo "Next: CUDA_TAG=${CUDA_TAG} bash tuned/smoke_test.sh ${GPU_TUNED_VARIANT}, review and commit ${NOTES}, then CUDA_TAG=${CUDA_TAG} bash tuned/release.sh ${GPU_TUNED_VARIANT}"
