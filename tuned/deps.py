"""Dependency lists for the tuned sglang wheels (zbrad/sglang tuned-builds).

Upstream's python/pyproject.toml is not edited. For a variant, tuned/repo.env names the upstream
dependencies to drop (GPU_TUNED_DROP_DEPS) and a requirements file whose lines replace them
(GPU_TUNED_REQUIREMENTS). This script applies that rule in the two places it is needed:

  python3 tuned/deps.py upstream <pyproject.toml> <drop names...>
      print upstream's [project] dependencies without the dropped ones (build.sh installs these)
  python3 tuned/deps.py rewrite <METADATA> <requirements.txt> <drop names...>
      remove the dropped names' Requires-Dist lines from a wheel's METADATA and add one
      Requires-Dist line per requirement (package.sh, on a staged copy)
"""

import logging
import re
import sys
import tomllib
from pathlib import Path
from typing import List, Set

logger = logging.getLogger(__name__)


class tuned_deps:
    """Filter and rewrite dependency lists by normalized distribution name."""

    _NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")

    def __init__(self, drop: List[str]) -> None:
        self._drop: Set[str] = {self.normalize(name) for name in drop}

    @staticmethod
    def normalize(name: str) -> str:
        """PEP 503 normalized name."""
        return re.sub(r"[-_.]+", "-", name).lower()

    def name_of(self, requirement: str) -> str:
        match = self._NAME.match(requirement)
        if match is None:
            raise ValueError(f"cannot read a distribution name from {requirement!r}")
        return self.normalize(match.group(1))

    def is_dropped(self, requirement: str) -> bool:
        return self.name_of(requirement) in self._drop

    def upstream(self, pyproject: Path) -> List[str]:
        with pyproject.open("rb") as handle:
            dependencies: List[str] = tomllib.load(handle)["project"]["dependencies"]
        kept = [dep for dep in dependencies if not self.is_dropped(dep)]
        found = {self.name_of(dep) for dep in dependencies}
        missing = sorted(self._drop - found)
        if missing:
            # A name upstream no longer lists means repo.env is out of date: say so, do not fail.
            logger.warning("dropped names not in %s: %s", pyproject, ", ".join(missing))
        return kept

    @staticmethod
    def requirement_lines(requirements: Path) -> List[str]:
        lines: List[str] = []
        for raw in requirements.read_text().splitlines():
            line = raw.split(" #", 1)[0].strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-"):
                raise ValueError(f"{requirements}: option lines ({line!r}) cannot become Requires-Dist")
            lines.append(line)
        return lines

    def rewrite(self, metadata: Path, requirements: Path) -> int:
        out: List[str] = []
        removed = 0
        added = self.requirement_lines(requirements)
        inserted = False
        for line in metadata.read_text().splitlines():
            if line.startswith("Requires-Dist: "):
                if self.is_dropped(line[len("Requires-Dist: "):]):
                    removed += 1
                    continue
                if not inserted:
                    out.extend(f"Requires-Dist: {req}" for req in added)
                    inserted = True
            out.append(line)
        if not inserted:
            # No surviving Requires-Dist line to anchor on: put them before the first blank line (end of headers).
            at = out.index("") if "" in out else len(out)
            out[at:at] = [f"Requires-Dist: {req}" for req in added]
        metadata.write_text("\n".join(out) + "\n")
        logger.info("%s: removed %d Requires-Dist, added %d", metadata, removed, len(added))
        return removed

    @classmethod
    def main(cls, argv: List[str]) -> int:
        logging.basicConfig(level=logging.INFO, format="deps.py: %(message)s", stream=sys.stderr)
        if len(argv) >= 2 and argv[0] == "upstream":
            for dep in cls(argv[2:]).upstream(Path(argv[1])):
                sys.stdout.write(dep + "\n")
            return 0
        if len(argv) >= 3 and argv[0] == "rewrite":
            cls(argv[3:]).rewrite(Path(argv[1]), Path(argv[2]))
            return 0
        sys.stderr.write(__doc__ or "")
        return 1


if __name__ == "__main__":
    sys.exit(tuned_deps.main(sys.argv[1:]))
