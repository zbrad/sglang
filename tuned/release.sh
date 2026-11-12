#!/bin/bash
# tuned/release.sh <variant> — run the release gates for the wheels tuned/package.sh made. Stage order:
#   build.sh -> full_test.sh -> package.sh -> smoke_test.sh -> (review and commit tuned/releases/RELEASE_NOTES_<version>.md) -> this -> publish.sh
# It never builds, tests, packages or publishes (docs/ARCHITECTURE.md B3, I5). If every gate passes it writes the
# ready-marker (gpu_tuned_release_ready, RELEASE_READY_<variant>); tuned/publish.sh <variant> publishes exactly what
# the gates saw. The gates are the common ones in tuned-common (gpu_tuned_release_gates), all reported together.
# As in zbrad/flashinfer, the gates run on the device-code wheel (sglang-kernel); the sglang wheel is attached beside it,
# and its sha256 was checked against the test log by package.sh and against its sidecar by smoke_test.sh.
#
#   CUDA_TAG=cu134 bash tuned/release.sh gb10
set -uo pipefail

REPODIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/release.sh <variant>}"

source "${REPODIR}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}" || exit 1
cd "${REPODIR}" || exit 1

INFO="${GPU_TUNED_DIST_DIR}/package-info.txt"
TEST_LOG="${GPU_TUNED_RELEASES_DIR}/TEST_RESULTS_${GPU_TUNED_VARIANT}.log"
SMOKE_LOG="${GPU_TUNED_RELEASES_DIR}/SMOKE_RESULTS_${GPU_TUNED_VARIANT}.log"
MARKER="${GPU_TUNED_RELEASES_DIR}/RELEASE_READY_${GPU_TUNED_VARIANT}"
gpu_tuned_release_marker_clear "${MARKER}"

if [[ ! -f "${INFO}" ]]; then
    echo "ERROR: ${INFO} not found -- run 'CUDA_TAG=${CUDA_TAG} bash tuned/package.sh ${GPU_TUNED_VARIANT}' at the commit being released first." >&2
    exit 1
fi
info() { sed -n "s/^$1=//p" "${INFO}"; }
RELEASE_TAG="$(info release_tag)"; PACKAGED_COMMIT="$(info commit)"; BASE_VERSION="$(info upstream_version)"
KERNEL_NAME="$(info wheel)"; KERNEL_WHEEL="${GPU_TUNED_DIST_DIR}/${KERNEL_NAME}"
SGLANG_NAME="$(info wheel_sglang)"; SGLANG_WHEEL="${GPU_TUNED_DIST_DIR}/${SGLANG_NAME}"
RELEASE_NOTES_FILE="tuned/releases/RELEASE_NOTES_${RELEASE_TAG#v}.md"
RELEASE_TITLE="sglang ${BASE_VERSION} — ${GPU_TUNED_VARIANT} ${CUDA_TAG} (${RELEASE_TAG#v}) — ${GPU_TUNED_HW_LABEL} wheels"

echo "Release ${RELEASE_TAG} (sglang ${BASE_VERSION}, packaged at ${PACKAGED_COMMIT}):"

gpu_tuned_release_gates tag="${RELEASE_TAG}" variant="${GPU_TUNED_VARIANT}" cuda="${CUDA_TAG}" info="${INFO}" \
    built="${GPU_TUNED_WHEEL_DIR}/${KERNEL_NAME}" test_log="${TEST_LOG}" \
    test_pass='^=== Full test suite summary: acceptance \+ tuned tests passed ===$' \
    tarball="${KERNEL_WHEEL}" smoke_log="${SMOKE_LOG}" notes="${RELEASE_NOTES_FILE}" packaged="${PACKAGED_COMMIT}" \
    || { echo "" >&2; echo "ERROR: a release gate failed; nothing was published (and no ready-marker was written)." >&2; exit 1; }
[[ "$(sha256sum "${SGLANG_WHEEL}" | awk '{print $1}')" == "$(info sha256_sglang)" ]] \
    || { echo "ERROR: ${SGLANG_NAME} changed since package.sh (sha256 differs from package-info.txt)." >&2; exit 1; }

echo ""
gpu_tuned_release_ready "${MARKER}" "zbrad/sglang" "${RELEASE_TAG}" "${RELEASE_TITLE}" \
    "${RELEASE_NOTES_FILE}" \
    "${KERNEL_WHEEL}#${KERNEL_NAME}" "${KERNEL_WHEEL}.sha256#${KERNEL_NAME}.sha256" \
    "${SGLANG_WHEEL}#${SGLANG_NAME}" "${SGLANG_WHEEL}.sha256#${SGLANG_NAME}.sha256" \
    "${TEST_LOG}#Test results (${GPU_TUNED_VARIANT})" \
    "${SMOKE_LOG}#Package smoke test results (${GPU_TUNED_VARIANT})" || exit 1

echo "Done."
