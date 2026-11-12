# Tuned builds of SGLang

Single-GPU, single-arch sglang wheels with no nvcc at run time, published as GitHub releases of this fork. The design
is `docs/ARCHITECTURE.md` in [zbrad/tuned-common](https://github.com/zbrad/tuned-common) (requirement IDs below refer to
it), and the scripts follow zbrad/vllm's `tuned/`. Shared logic is vendored as `tuned/common.sh`, hardware and toolkit
facts as `tuned/common-data/`; keep both current with `tuned/sync-common.sh check|sync`. Repo settings are in
`tuned/repo.env`. Only `gb10` (DGX Spark, sm_121a) with `cu134` is set up. **Status: scripts written, not yet built.**

## Flow

```
CUDA_TAG=cu134 bash tuned/build.sh gb10        # venv, dependencies, both wheels, then seed + prove the JIT cache
CUDA_TAG=cu134 bash tuned/full_test.sh gb10    # gates, strict acceptance run, tuned/tests
CUDA_TAG=cu134 bash tuned/package.sh gb10      # stamp, rewrite metadata, add the cache, sha256, package-info, DRAFT notes
CUDA_TAG=cu134 bash tuned/smoke_test.sh gb10   # the packaged wheels as a consumer gets them
# review tuned/releases/RELEASE_NOTES_<version>.md, remove the DRAFT banner, commit and push
CUDA_TAG=cu134 bash tuned/release.sh gb10      # every gate together; on pass writes RELEASE_READY_gb10 (publishes nothing)
CUDA_TAG=cu134 bash tuned/publish.sh gb10      # publish exactly what release.sh's gates saw
```

`tuned/wheel.sh <variant>` is the wheel step `build.sh` calls; it refuses without the build venv. One of `CUDA_TAG` or
`CUDA_VER` is required: there is no default toolkit.

Host prerequisites: the CUDA 13.4 toolkit at `/usr/local/cuda-13.4`, and Rust for dependencies with no cp314 wheel
(`sudo apt-get install rustc-1.97 cargo-1.97`; `outlines_core==0.1.26` is built from source).

## What is built

| Wheel | From | Notes |
|---|---|---|
| `sglang-kernel` | `python/sglang/kernels/aot` (scikit-build-core) | `-DSGL_KERNEL_TUNED_ARCH=121a`: SASS for sm_121a only, no PTX; the sm90 copy and FA3 are not built. Re-versioned to `<version>+gb10.cu134.<label>`. |
| `sglang` | `python/` (setuptools-scm) | version `<public>+gb10.cu134.<label>`; carries the seeded JIT cache and `sglang_tuned.pth` (packaged wheel only). |

The `SGL_KERNEL_TUNED_ARCH` switch in `python/sglang/kernels/aot/CMakeLists.txt` is the only change to an upstream file;
unset, the build is upstream's.

## No JIT at run time

- **sglang-kernel**: one arch, SASS only, checked on every device-code `.so` (`gpu_tuned_verify_arch`, `cuobjdump --list-ptx`).
- **sglang's own JIT kernels** (`python/sglang/kernels/jit`, compiled lazily with nvcc and keyed by per-call template
  arguments): `build.sh` seeds `SGLANG_JIT_CACHE_DIR` by running the acceptance models once, then runs them again with
  `SGLANG_CRASH_ON_JIT_COMPILE=1`, where any cache miss is an error. The packaged wheel ships that cache and a
  `sglang_tuned.pth` that sets, with `os.environ.setdefault` (so the environment still wins), `SGLANG_JIT_CACHE_DIR` to
  it, `SGLANG_CRASH_ON_JIT_COMPILE=1`, `FLASHINFER_DISABLE_JIT=1` and `SGLANG_ENABLE_JIT_DEEPGEMM=0`.
  The cache only covers the module shapes the acceptance models reach: another model may need a kernel that is not
  there, and then fails rather than compiling.
- **Consumer requirement**: the cache key fingerprints `nvcc --version`, `c++ --version` and some package versions
  (`_environment_fingerprint` in `python/sglang/kernels/jit/utils/compile/cache.py`), so the host needs the same CUDA 13.4
  nvcc and host compiler as the build (nvcc is asked its version, never run as a compiler), or every lookup misses.
- **flashinfer**: the zbrad/flashinfer AOT jit-cache wheels, `FLASHINFER_DISABLE_JIT=1`.
- **Not covered**: Triton compiles its own kernels at run time (as in zbrad/vllm).

## Dependencies

Upstream's `python/pyproject.toml` is not edited. `tuned/repo.env` lists what gb10 drops (`GPU_TUNED_DROP_DEPS`, with a
reason for each), and `tuned/requirements-gb10.txt` what replaces it (tuned torch 2.15.0, flashinfer 0.7.4 and torchvision 0.29.2 by URL).
`build.sh` installs that file, guards the torch pin (`gpu_tuned_protect_torch_pin`), then installs upstream's remaining
dependencies (`tuned/deps.py upstream`). `package.sh` rewrites the packaged wheels' `Requires-Dist` the same way
(`tuned/deps.py rewrite`), adding `sglang-kernel @ <this release's URL>`. Dropped for gb10: `sgl-flash-mla` (no sm_12x
kernels), `sgl-deep-gemm` (JIT-only), `sgl-deep-ep` (multi-GPU), `outlines` (optional grammar backend; its
`outlines_core` has no cp314 wheel and its PyO3 0.22 refuses Python 3.14, so `--grammar-backend outlines` is not
available; xgrammar and llguidance are), `torchaudio` (no build for the tuned torch, so audio models are out of
scope). `torchvision` is replaced by the zbrad/vision tuned wheel (sglang imports it at start-up), built without image codecs.

## Where things are

Everything a build makes is in `tuned/build/<cuda-tag>/` (ignored by git, one tree per toolkit, C2): `venv/`, `cmake/`
(both CMake trees and the re-version staging), `ccache/`, `jit-cache/` (the seeded cache), `wheel/` (unstamped wheels
that `full_test.sh` tests), `dist/` (packaged wheels, sidecars, `package-info.txt`), `releases/` (test and smoke logs,
`RELEASE_READY_<variant>`), and the markers `release-label`, `build-commit`, `jit-cache-seeded`.

## Test scope (G3, recorded in compliance.conf)

`full_test.sh` runs the gates, `tuned/gb10_acceptance.py strict` (NVIDIA-Nemotron-3-Nano-4B-BF16 and
Nano-30B-A3B-NVFP4, the stand-ins zbrad/vllm uses; needs a Hugging Face download) and `pytest tuned/tests`. Upstream's
`test/` suite and multi-GPU tests are not run.

## Upstream's agent files

Upstream's agent rule files are in `rules_advisory/` (upstream's, not followed by these scripts).
