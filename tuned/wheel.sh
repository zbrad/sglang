#!/bin/bash
# tuned/wheel.sh <variant> — make the unstamped sglang-kernel and sglang wheels for one GPU variant and one CUDA
# toolkit, into tuned/build/<cuda-tag>/wheel/, and record the release label and the build commit. This is the wheel
# step of tuned/build.sh, which installs the toolkit's build venv first; it refuses to run without that venv. It never
# tests, packages or publishes (package.sh stamps a copy, H3).
#
#   CUDA_TAG=cu134 bash tuned/wheel.sh gb10     (normally run by tuned/build.sh)
#
# The release label (E3/E4) is allocated here, once, and both wheels carry it as their local version:
#   sglang-kernel  <its pyproject version>+<variant>.cu<cuda>.<label>   (re-versioned after the build: the version is static)
#   sglang         <setuptools-scm public version>+<variant>.cu<cuda>.<label>   (SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG)
# sglang-kernel is built single-arch, SASS only (-DSGL_KERNEL_TUNED_ARCH, zbrad/sglang tuned-builds CMake edit).
# release-label and build-commit are written last, so their presence means the build finished.
set -euo pipefail

GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/wheel.sh <variant>}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=env.sh
source "${REPO_ROOT}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}"

[[ -d "${GPU_TUNED_VENV_DIR}" ]] || {
    echo "ERROR: ${GPU_TUNED_VENV_DIR} not found. Run CUDA_TAG=${CUDA_TAG} bash tuned/build.sh ${GPU_TUNED_VARIANT} (it installs the build venv, then calls this script)." >&2
    exit 1
}
gpu_tuned_verify_venv "${GPU_TUNED_VENV_DIR}" "${REPO_ROOT}"
# shellcheck source=/dev/null
source "${GPU_TUNED_VENV_DIR}/bin/activate"

LOCAL_VERSION="$(gpu_tuned_build_local_version "${GPU_TUNED_VARIANT}" "${REPO_ROOT}")" || exit 1
[[ "${LOCAL_VERSION}" == *".cu${CUDA_VERSION_COMPACT}."* ]] || {
    echo "ERROR: label ${LOCAL_VERSION} disagrees with ${CUDA_TAG} (from ${CUDA_HOME}); nvcc and the toolkit definition must report the same release." >&2
    exit 1
}
LABEL="${LOCAL_VERSION#*.cu${CUDA_VERSION_COMPACT}.}"
echo "Release label: ${LABEL} (local version ${LOCAL_VERSION})"

# Remove the previous build's markers first: they are rewritten only if this build succeeds.
rm -f "${GPU_TUNED_BUILD_ROOT}/build-commit" "${GPU_TUNED_BUILD_ROOT}/release-label"
mkdir -p "${GPU_TUNED_WHEEL_DIR}" "${GPU_TUNED_CMAKE_DIR}"
rm -f "${GPU_TUNED_WHEEL_DIR}"/sglang_kernel-*.whl "${GPU_TUNED_WHEEL_DIR}"/sglang-*.whl

echo "=========================================="
echo "Building sglang-kernel for ${GPU_TUNED_HW_LABEL}"
echo "=========================================="
echo "Arch:             sm_${GPU_TUNED_CUDA_ARCH} only, SASS (no PTX)"
echo "CUDA_HOME:        ${CUDA_HOME}"
echo "Parallel:         CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL}, nvcc --threads=${NVCC_THREADS}"
echo "CMake build tree: ${GPU_TUNED_CMAKE_DIR}/sglang-kernel"
echo "Git commit:       $(git rev-parse --short HEAD)"

# --no-isolation: the venv holds the tuned torch (pyproject's torch==2.14.1 build pin is upstream's, so the
# dependency check is skipped). build-dir keeps CMake out of the source tree and per toolkit (C1/C6).
KERNEL_RAW_DIR="${GPU_TUNED_CMAKE_DIR}/sglang-kernel-wheel"
rm -rf "${KERNEL_RAW_DIR}"
python3 -m build --wheel --no-isolation --skip-dependency-check --outdir "${KERNEL_RAW_DIR}" \
    -C "build-dir=${GPU_TUNED_CMAKE_DIR}/sglang-kernel" \
    -C "cmake.define.SGL_KERNEL_TUNED_ARCH=${GPU_TUNED_CUDA_ARCH}" \
    -C "cmake.define.SGL_KERNEL_ENABLE_FA3=OFF" \
    -C "cmake.define.ENABLE_BELOW_SM90=OFF" \
    -C "cmake.define.SGL_KERNEL_COMPILE_THREADS=${NVCC_THREADS}" \
    -C "cmake.define.ENABLE_CCACHE=ON" \
    python/sglang/kernels/aot

RAW_KERNEL="$(ls "${KERNEL_RAW_DIR}"/sglang_kernel-*.whl 2>/dev/null | head -1)"
[[ -n "${RAW_KERNEL}" ]] || { echo "ERROR: no sglang_kernel wheel in ${KERNEL_RAW_DIR}" >&2; exit 1; }
KERNEL_PUBLIC="$(gpu_tuned_wheel_version "${RAW_KERNEL}" sglang_kernel)" || exit 1
[[ "${KERNEL_PUBLIC}" != *+* ]] || { echo "ERROR: ${RAW_KERNEL##*/} already has a local version (${KERNEL_PUBLIC})." >&2; exit 1; }

# Re-version: the dist-info directory name is what `wheel pack` names the wheel by; METADATA's Version must match.
KERNEL_VERSION="${KERNEL_PUBLIC}+${LOCAL_VERSION}"
UNPACK_DIR="${GPU_TUNED_CMAKE_DIR}/sglang-kernel-reversion"
rm -rf "${UNPACK_DIR}"
python3 -m wheel unpack "${RAW_KERNEL}" --dest "${UNPACK_DIR}"
UNPACKED="$(ls -d "${UNPACK_DIR}"/sglang_kernel-*)"
DIST_INFO="$(ls -d "${UNPACKED}"/sglang_kernel-*.dist-info)"
sed -i "s/^Version: ${KERNEL_PUBLIC}\$/Version: ${KERNEL_VERSION}/" "${DIST_INFO}/METADATA"
grep -qx "Version: ${KERNEL_VERSION}" "${DIST_INFO}/METADATA" || { echo "ERROR: could not set Version in ${DIST_INFO}/METADATA" >&2; exit 1; }
mv "${DIST_INFO}" "${UNPACKED}/sglang_kernel-${KERNEL_VERSION}.dist-info"
python3 -m wheel pack "${UNPACKED}" --dest-dir "${GPU_TUNED_WHEEL_DIR}"
KERNEL_WHEEL="$(ls "${GPU_TUNED_WHEEL_DIR}"/sglang_kernel-*.whl | head -1)"

# Single arch, SASS only, in every shared object that carries device code (the gate package.sh repeats).
KERNEL_SOS=0
while IFS= read -r so; do
    gpu_tuned_so_has_device_code "${so}" || continue
    gpu_tuned_verify_arch "${so}" "${GPU_TUNED_CUDA_ARCH}" || { echo "ERROR: ${so#"${UNPACKED}"/} is not sm_${GPU_TUNED_CUDA_ARCH} only." >&2; exit 1; }
    KERNEL_SOS=$((KERNEL_SOS + 1))
done < <(find "${UNPACKED}" -name '*.so')
[[ "${KERNEL_SOS}" -gt 0 ]] || { echo "ERROR: no shared object with device code in ${KERNEL_WHEEL##*/}." >&2; exit 1; }
# No shared object may depend on a stock torch library name: the tuned torch ships only -tuned names, so a stock NEEDED
# entry would resolve to whatever stock copy the loader finds first (or nothing).
STOCK_NEEDED_RE='^lib(c10|c10_cuda|torch|torch_cpu|torch_cuda|torch_cuda_linalg|torch_python|torch_global_deps|caffe2_nvrtc|shm)\.so(\.[0-9]+)*$'
while IFS= read -r so; do
    stock="$(readelf -d "${so}" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p' | grep -E "${STOCK_NEEDED_RE}" || true)"
    [[ -z "${stock}" ]] || { echo "ERROR: ${so#"${UNPACKED}"/} NEEDs stock torch libraries: $(echo ${stock})" >&2; exit 1; }
done < <(find "${UNPACKED}" -name '*.so')
echo "OK: no sglang-kernel shared object NEEDs a stock torch library name"
[[ ! -d "${UNPACKED}/sgl_kernel/sm90" ]] || { echo "ERROR: ${KERNEL_WHEEL##*/} still ships sgl_kernel/sm90 (SGL_KERNEL_TUNED_ARCH not applied?)." >&2; exit 1; }
echo "OK: ${KERNEL_SOS} sglang-kernel shared objects are sm_${GPU_TUNED_CUDA_ARCH} only"

echo "=========================================="
echo "Building sglang for ${GPU_TUNED_HW_LABEL}"
echo "=========================================="
# setuptools-scm's public version for this commit (upstream's describe command), then our local version on it.
SGLANG_PUBLIC="$(cd python && python3 -m setuptools_scm)"
SGLANG_PUBLIC="${SGLANG_PUBLIC%%+*}"
[[ -n "${SGLANG_PUBLIC}" && "${SGLANG_PUBLIC}" != 0.0.0* ]] || { echo "ERROR: setuptools-scm gave no usable version for python/ ('${SGLANG_PUBLIC}')." >&2; exit 1; }
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG="${SGLANG_PUBLIC}+${LOCAL_VERSION}"
echo "Version: ${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG}"
# --build-base keeps setuptools' build/ tree out of python/ (C6): left there, a later build packaged the earlier one's
# build/lib/sglang into the wheel a second time (40 MB instead of 21 MB, found 2026-11-12).
python3 -m build --wheel --no-isolation --skip-dependency-check --outdir "${GPU_TUNED_WHEEL_DIR}" \
    --config-setting="--build-option=build --build-base=${GPU_TUNED_CMAKE_DIR}/sglang-py" python
[[ ! -e python/build ]] || { echo "ERROR: the sglang build wrote python/build despite --build-base." >&2; exit 1; }

SGLANG_WHEEL="$(ls "${GPU_TUNED_WHEEL_DIR}"/sglang-*.whl 2>/dev/null | head -1)"
[[ -n "${SGLANG_WHEEL}" ]] || { echo "ERROR: no sglang wheel found in ${GPU_TUNED_WHEEL_DIR}" >&2; exit 1; }
STRAY="$(python3 -c 'import sys, zipfile; print(sum(n.startswith("build/") for n in zipfile.ZipFile(sys.argv[1]).namelist()))' "${SGLANG_WHEEL}")"
[[ "${STRAY}" -eq 0 ]] || { echo "ERROR: ${SGLANG_WHEEL##*/} contains ${STRAY} entries under build/ (a stale setuptools tree)." >&2; exit 1; }
for w in "${KERNEL_WHEEL}" "${SGLANG_WHEEL}"; do
    case "${w##*/}" in sglang_kernel-*) pkg=sglang_kernel ;; *) pkg=sglang ;; esac
    v="$(gpu_tuned_wheel_version "${w}" "${pkg}")" || exit 1
    [[ "${v}" == *"+${GPU_TUNED_VARIANT}.cu${CUDA_VERSION_COMPACT}.${LABEL}" ]] || {
        echo "ERROR: ${w##*/} is version ${v}, not carrying +${GPU_TUNED_VARIANT}.cu${CUDA_VERSION_COMPACT}.${LABEL} from the release label." >&2
        exit 1
    }
    gpu_tuned_check_wheel_label "${w}"
    echo "Built wheel: ${w##*/} ($(du -sh "${w}" | awk '{print $1}'))"
done

# build-commit: the source the wheels were made from. A dirty tree is recorded as such and package.sh refuses it.
BUILD_COMMIT="$(git rev-parse HEAD)"
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || BUILD_COMMIT="${BUILD_COMMIT}-dirty"
printf '%s\n' "${LABEL}" > "${GPU_TUNED_BUILD_ROOT}/release-label"
printf '%s\n' "${BUILD_COMMIT}" > "${GPU_TUNED_BUILD_ROOT}/build-commit"

echo ""
echo "Wheels:       ${KERNEL_WHEEL##*/} ${SGLANG_WHEEL##*/}"
echo "Label:        ${LABEL}"
echo "Build commit: ${BUILD_COMMIT}"
