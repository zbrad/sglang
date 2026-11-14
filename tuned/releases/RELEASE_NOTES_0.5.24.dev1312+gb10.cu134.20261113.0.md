# sglang 0.5.24.dev1312 — gb10 cu134 (0.5.24.dev1312+gb10.cu134.20261113.0)

Single-GPU sglang for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated), CUDA 13.4: sglang-kernel built for sm_121a only
(SASS, no PTX), and no nvcc at run time (sglang's JIT kernels come from a cache seeded at build time).

- Release label: 20261113.0 (UTC date of the tuned commit)
- Built from commit 2e4f01f054b4dc77ffae08be804e5050548893a2; tested at a81a4c839165379c73c73361f4899eec1a95c19c; packaged at 018e1231712c0dfdefbc67fdda1e930e9071290c (later commits change only tuned/)
- Upstream base: upstream/main merge-base 993199e9fcb9e333ffb045b2c6ccdcc944e43e26
- Dependencies: tuned/requirements-gb10.txt; dropped from upstream's list: torch flashinfer_python sglang-kernel sgl-flash-mla sgl-deep-gemm sgl-deep-ep torchvision torchaudio outlines pillow (reasons in tuned/repo.env)
- Seeded JIT modules: 6
- sha256: sglang_kernel-0.4.10+gb10.cu134.20261113.0-cp311-abi3-linux_aarch64.whl f55cc7dccc31f1d1cb9a62b5f7995ed52f496187292f50b172b89d8d6c5d5d35; sglang-0.5.24.dev1312+gb10.cu134.20261113.0-py3-none-any.whl 50125120f819f434aac288a0f9d51bead3a6ab2dd467b98a4312a822eb17da18

## Pinned dependencies (tuned/requirements-gb10.txt)

- Python 3.14 (the torch wheel is cp314)
- torch 2.15.0+gb10.cu134.20261030.1 (zbrad/pytorch; upstream sglang pins 2.14.1)
- flashinfer-python, flashinfer-jit-cache, flashinfer-jit-cache-sm121a 0.7.4+gb10.cu134.20261113.1 (zbrad/flashinfer;
  that release fixes `fp4_quantize` asking for a module its cache does not ship, which failed NVFP4 models under
  `FLASHINFER_DISABLE_JIT=1`)
- torchvision 0.29.2+gb10.cu134.20261112.0 (zbrad/vision; no image codecs)
- pillow>=13 in place of upstream's `pillow<13`: pillow 12.3.0 has advisories fixed only in 13.0.0. Upstream caps it
  because its Rust image pipeline reproduces Pillow 12.x bicubic resizing; image models were not tested with 13.

## Strict no-JIT by default

The wheel installs `sglang_tuned.pth`, which sets, unless already set: `SGLANG_JIT_CACHE_DIR` (the shipped cache),
`SGLANG_CRASH_ON_JIT_COMPILE=1`, `FLASHINFER_DISABLE_JIT=1`, `SGLANG_ENABLE_JIT_DEEPGEMM=0`. A kernel missing from a
cache is an error, not a compile.

## Known issue: slow weight loading under the strict settings

Measured on one GB10 node, weights on a single-HDD ZFS pool with an NVMe L2ARC (shard-loading time as sglang reports it):

| model | seed run (JIT allowed, empty sglang cache) | strict run (run 1 / run 2) |
|---|---|---|
| NVIDIA-Nemotron-3-Nano-4B-BF16 | 3:26 | 19:49 / 18:53 |
| NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 | 1:28 | 1:03:30 / 52:11 |

During the slow loads one scheduler thread is CPU-bound in `param_data.copy_(loaded_weight)` inside the
linear-layer weight loaders, with no disk reads. Setting `SGLANG_CRASH_ON_JIT_COMPILE=1` alone or
`FLASHINFER_DISABLE_JIT=1` alone reproduced it (4B: 984 s and 964 s to engine ready), so neither flag on its own is
the cause. Not yet explained; serving speed after the load was not measured.

## Runtime requirements

sglang's JIT cache key fingerprints `nvcc --version` and `c++ --version`: the host needs the CUDA 13.4 toolkit
(nvcc is asked its version, never run as a compiler) and the same host compiler as the build, or every lookup misses
and, with SGLANG_CRASH_ON_JIT_COMPILE=1, fails.

## Tests

Scope (from the attached test log): gates; tuned/gb10_acceptance.py strict (stand-ins); python -m pytest tuned/tests
Skips by design: upstream's test/ suite; multi-GPU; Triton's own runtime compiles (out of the no-JIT scope)
Result: === Full test suite summary: acceptance + tuned tests passed ===

## Install

```
pip install "sglang @ https://github.com/zbrad/sglang/releases/download/v0.5.24.dev1312%2Bgb10.cu134.20261113.0/sglang-0.5.24.dev1312+gb10.cu134.20261113.0-py3-none-any.whl"
```
