"""Run bookkeeping.

Every experiment writes into its own output directory, and that directory
records enough to reproduce itself. Call `start_run()` first thing in `main()`.
"""

from __future__ import annotations

import json
import platform
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
OUTPUTS = ROOT / "outputs"


def load_config(name: str) -> dict[str, Any]:
    """Read `configs/<name>.yaml`.

    Config files hold parameters only. Anything that branches, inherits or
    computes belongs in Python.
    """
    path = CONFIGS / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no config at {path}")
    with path.open() as f:
        return yaml.safe_load(f) or {}


def start_run(exp_id: str, config: dict[str, Any]) -> Path:
    """Create `outputs/<exp_id>/`, record provenance, and return the directory.

    Writes `config.yaml` (exactly the parameters used) and `metadata.json`
    (commit, dirty flag, timestamp, seed, interpreter, command line).
    """
    out = OUTPUTS / exp_id
    out.mkdir(parents=True, exist_ok=True)

    with (out / "config.yaml").open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    porcelain = _git("status", "--porcelain")
    metadata = {
        "exp_id": exp_id,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": shlex.join(sys.argv),
        "seed": config.get("seed"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": None if porcelain is None else porcelain != "",
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    with (out / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    return out


def _git(*args: str) -> str | None:
    """Run a git command in the repo root; None if git or the repo is absent."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None
