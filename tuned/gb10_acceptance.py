"""Acceptance test for the tuned sglang wheels on one GPU (zbrad/sglang tuned-builds).

  python -I tuned/gb10_acceptance.py <seed|strict> [model ...]

Loads each stand-in model with sglang's offline Engine and generates a few tokens.

  seed    runtime JIT allowed: sglang's JIT kernels compile into SGLANG_JIT_CACHE_DIR (the build seeds
          the cache this way, from exactly these models).
  strict  nothing may compile with nvcc at run time: SGLANG_CRASH_ON_JIT_COMPILE=1 (a sglang JIT kernel
          missing from the cache is an error), FLASHINFER_DISABLE_JIT=1 (only flashinfer's AOT jit-cache),
          SGLANG_ENABLE_JIT_DEEPGEMM=0. Triton still compiles its own kernels (out of scope, as in vllm).

The models are small stand-ins (the same ones zbrad/vllm's tuned/gb10_acceptance_test.sh runs), not the
target models. Each runs in its own process so one failure does not hide the others. Exit status 0 only
if every model loaded and produced non-empty text.
"""

import logging
import os
import subprocess
import sys
from typing import Dict, List

logger = logging.getLogger(__name__)


class gb10_acceptance:
    """Run each stand-in model in a child process under the chosen JIT mode."""

    MODELS: List[str] = [
        "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",  # NemotronH dense, stand-in for Nemotron-3-Super-120B
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",  # NemotronH NVFP4 MoE + mamba
    ]
    STRICT_ENV: Dict[str, str] = {
        "SGLANG_CRASH_ON_JIT_COMPILE": "1",
        "FLASHINFER_DISABLE_JIT": "1",
        "SGLANG_ENABLE_JIT_DEEPGEMM": "0",
    }
    PROMPT = "The capital of France is"
    CHILD = """
import sys
import sglang
engine = sglang.Engine(model_path=sys.argv[1], mem_fraction_static=float(sys.argv[2]), context_length=4096,
                       log_level="warning")
try:
    out = engine.generate(sys.argv[3], {"max_new_tokens": 16, "temperature": 0.0})
    text = out["text"] if isinstance(out, dict) else out[0]["text"]
    print("GENERATED: " + repr(text))
    if not text.strip():
        raise SystemExit("FAIL: empty generation")
finally:
    engine.shutdown()
"""

    def __init__(self, mode: str, models: List[str]) -> None:
        if mode not in ("seed", "strict"):
            raise ValueError(f"mode must be seed or strict, not {mode!r}")
        self._mode = mode
        self._models = models or self.MODELS
        self._mem_fraction = os.environ.get("ACCEPT_MEM_FRACTION", "0.30")

    def child_env(self) -> Dict[str, str]:
        env = dict(os.environ)
        if self._mode == "strict":
            env.update(self.STRICT_ENV)
        else:
            for key in self.STRICT_ENV:
                env.pop(key, None)
            env["SGLANG_ENABLE_JIT_DEEPGEMM"] = "0"  # deep_gemm is not shipped: never seed for it
        return env

    def run_one(self, model: str) -> bool:
        logger.info("=== %s (%s) ===", model, self._mode)
        result = subprocess.run(
            [sys.executable, "-I", "-c", self.CHILD, model, self._mem_fraction, self.PROMPT],
            env=self.child_env(),
        )
        ok = result.returncode == 0
        logger.info("%s: %s", "PASS" if ok else "FAIL", model)
        return ok

    def run(self) -> int:
        cache = os.environ.get("SGLANG_JIT_CACHE_DIR", "")
        if not cache:
            raise ValueError("SGLANG_JIT_CACHE_DIR must be set (tuned/env.sh sets it)")
        logger.info("mode=%s SGLANG_JIT_CACHE_DIR=%s", self._mode, cache)
        failed = [model for model in self._models if not self.run_one(model)]
        if failed:
            logger.error("FAILED: %s", ", ".join(failed))
            return 1
        logger.info("all %d models passed (%s)", len(self._models), self._mode)
        return 0

    @classmethod
    def main(cls, argv: List[str]) -> int:
        logging.basicConfig(level=logging.INFO, format="gb10_acceptance: %(message)s", stream=sys.stderr)
        if not argv:
            sys.stderr.write(__doc__ or "")
            return 2
        return cls(argv[0], argv[1:]).run()


if __name__ == "__main__":
    sys.exit(gb10_acceptance.main(sys.argv[1:]))
