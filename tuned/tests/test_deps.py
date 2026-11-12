"""Tests for tuned/deps.py (run: python3 -m pytest tuned/tests/test_deps.py)."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "tuned_deps_module", Path(__file__).resolve().parents[1] / "deps.py"
)
assert _SPEC is not None and _SPEC.loader is not None
deps: ModuleType = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(deps)

METADATA = """Metadata-Version: 2.4
Name: sglang
Version: 0.5.24+gb10.cu134.20261112
Requires-Dist: aiohttp
Requires-Dist: torch==2.14.1
Requires-Dist: flashinfer_python[cu13]==0.7.4
Requires-Dist: sgl-flash-mla==1.0.4
Requires-Dist: pytest; extra == "test"

Long description.
"""


def test_upstream_drops_by_normalized_name(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\ndependencies = ["aiohttp", "torch==2.14.1", "Flashinfer-Python[cu13]==0.7.4"]\n'
    )
    kept = deps.tuned_deps(["torch", "flashinfer_python"]).upstream(pyproject)
    assert kept == ["aiohttp"]


def test_rewrite_replaces_dropped_requires_dist(tmp_path: Path) -> None:
    metadata = tmp_path / "METADATA"
    metadata.write_text(METADATA)
    requirements = tmp_path / "req.txt"
    requirements.write_text("# comment\ntorch @ https://example.invalid/torch.whl  # tuned\n")
    removed = deps.tuned_deps(["torch", "flashinfer_python", "sgl-flash-mla"]).rewrite(
        metadata, requirements
    )
    text = metadata.read_text()
    assert removed == 3
    assert "Requires-Dist: torch @ https://example.invalid/torch.whl" in text
    assert "torch==2.14.1" not in text and "flash-mla" not in text and "flashinfer" not in text
    assert "Requires-Dist: aiohttp" in text and 'extra == "test"' in text
    assert text.endswith("Long description.\n")


def test_rewrite_option_line_raises(tmp_path: Path) -> None:
    metadata = tmp_path / "METADATA"
    metadata.write_text(METADATA)
    requirements = tmp_path / "req.txt"
    requirements.write_text("-r other.txt\n")
    with pytest.raises(ValueError):
        deps.tuned_deps(["torch"]).rewrite(metadata, requirements)
