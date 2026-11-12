#!/bin/bash
# tuned/publish.sh <variant> — publish the GitHub release that tuned/release.sh marked ready. It publishes exactly what
# the gates saw: it refuses unless the marker exists and HEAD, the notes and every asset are unchanged since release.sh
# ran (gpu_tuned_publish_ready). It builds, tests and packages nothing.
#
#   CUDA_TAG=cu134 bash tuned/publish.sh gb10
set -uo pipefail

REPODIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_TUNED_ARG_VARIANT="${1:?usage: tuned/publish.sh <variant>}"

# shellcheck source=env.sh
source "${REPODIR}/tuned/env.sh" "${GPU_TUNED_ARG_VARIANT}" || exit 1
cd "${REPODIR}" || exit 1

MARKER="${GPU_TUNED_RELEASES_DIR}/RELEASE_READY_${GPU_TUNED_VARIANT}"
TAG="$(sed -n 's/^tag=//p' "${MARKER}" 2>/dev/null)"
gpu_tuned_publish_ready "${MARKER}" || exit 1

echo ""
echo "Release: https://github.com/zbrad/sglang/releases/tag/${TAG}"
echo "Done."
