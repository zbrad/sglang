# sglang 0.5.24.dev1319 — gb10 cu134 (0.5.24.dev1319+gb10.cu134.20261115.0)

Single-GPU sglang for NVIDIA GB10 (DGX Spark, Blackwell, SM 12.1 accelerated), CUDA 13.4: sglang-kernel built for sm_121a only
(SASS, no PTX), and no nvcc at run time (sglang's JIT kernels come from a cache seeded at build time).

- Release label: 20261115.0 (UTC date of the tuned commit)
- Built from commit 75758aa813aafafc9baaa15a632d81e56b60cbd8; tested at 75758aa813aafafc9baaa15a632d81e56b60cbd8
- Upstream base: upstream/main merge-base 993199e9fcb9e333ffb045b2c6ccdcc944e43e26
- Dependencies: tuned/requirements-gb10.txt; dropped from upstream's list: torch flashinfer_python sglang-kernel sgl-flash-mla sgl-deep-gemm sgl-deep-ep torchvision torchaudio outlines pillow (reasons in tuned/repo.env)
- Seeded JIT modules: 6
- sha256: sglang_kernel-0.4.10+gb10.cu134.20261115.0-cp311-abi3-linux_aarch64.whl 33a1367bc9004bd279eabd4cd70052f79c39cb73ebb166bf0370e1ac50922bf4; sglang-0.5.24.dev1319+gb10.cu134.20261115.0-py3-none-any.whl c2a6a0c1401c97392b6513043942e9f67cd931fdb6f043df312364e54c779705

## Changes since 0.5.24.dev1312+gb10.cu134.20261113.0

- **Weight loading is fast on GB10 (`75758aa813`).** Cherry-pick of upstream's open PR sgl-project/sglang#46670
  (commit `16a974d868`, Guanxin Li): on integrated GPUs both safetensors iterators touch one byte per page of each
  mmap view before handing it out. The pages stay file-backed, so no extra host memory is used. Discrete GPUs are
  unchanged. Drop this commit once upstream merges the PR.
- The previous release's known issue (strict-mode loads of 19 min to over an hour) is explained and fixed by the
  above. Neither the strict settings nor the shipped JIT cache was the cause: on GB10 the first host-to-GPU copy of an
  untouched safetensors mmap view runs at 3-5 MB/s, and a second copy of the same tensor at 17-55 GB/s. Measured on
  driver R615 and R620 alike, and with safetensors 0.7.0 and 0.8.0. vLLM has the same fix
  (vllm-project/vllm#58868, issue #58726).
- Commits `9eba754084` (weight_loader_disable_mmap on by default) and its revert `11443f5142` cancel out; that
  workaround was replaced by the fix above.

## Pinned dependencies (tuned/requirements-gb10.txt)

- Python 3.14 (the torch wheel is cp314)
- torch 2.15.0+gb10.cu134.20261030.1 (zbrad/pytorch; upstream sglang pins 2.14.1)
- flashinfer-python, flashinfer-jit-cache, flashinfer-jit-cache-sm121a 0.7.4+gb10.cu134.20261113.1 (zbrad/flashinfer)
- torchvision 0.29.2+gb10.cu134.20261112.0 (zbrad/vision; no image codecs)
- pillow>=13 in place of upstream's `pillow<13`: pillow 12.3.0 has advisories fixed only in 13.0.0. Upstream caps it
  because its Rust image pipeline reproduces Pillow 12.x bicubic resizing; image models were not tested with 13.

## Strict no-JIT by default

The wheel installs `sglang_tuned.pth`, which sets, unless already set: `SGLANG_JIT_CACHE_DIR` (the shipped cache),
`SGLANG_CRASH_ON_JIT_COMPILE=1`, `FLASHINFER_DISABLE_JIT=1`, `SGLANG_ENABLE_JIT_DEEPGEMM=0`. A kernel missing from a
cache is an error, not a compile.

## Weight-loading times

One GB10 node, weights on a single-HDD ZFS pool with an NVMe L2ARC; shard-loading time as sglang reports it. Times
depend on whether the shards are already in the page cache (the longer of the new numbers are likely reads from the disk; not verified).

| model | previous release, strict | this release, strict (full test / smoke test) |
|---|---|---|
| NVIDIA-Nemotron-3-Nano-4B-BF16 | 18:53-19:49 | 0:00 / 0:00 |
| NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 | 52:11-1:03:30 | 0:05 / 2:26 |

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
pip install "sglang @ https://github.com/zbrad/sglang/releases/download/v0.5.24.dev1319%2Bgb10.cu134.20261115.0/sglang-0.5.24.dev1319+gb10.cu134.20261115.0-py3-none-any.whl"
```
