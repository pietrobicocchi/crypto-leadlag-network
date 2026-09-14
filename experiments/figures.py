"""Promote experiment outputs into the paper.

`FIGURES` maps a paper figure to the experiment output that produced it. It is
the answer to "what produced Figure 2?", so keep it accurate: a figure becomes
a paper figure by being added here, and by nothing else.

    make figures
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
PAPER_FIGURES = ROOT / "paper" / "figures"

# paper figure name  ->  path under outputs/
FIGURES: dict[str, str] = {}


def main() -> None:
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    missing = []

    for name, source in FIGURES.items():
        path = OUTPUTS / source
        if not path.exists():
            missing.append(source)
            continue
        shutil.copy2(path, PAPER_FIGURES / name)
        print(f"{source}  ->  paper/figures/{name}")

    if missing:
        print("\nmissing outputs (run the experiment that produces them):", file=sys.stderr)
        for source in missing:
            print(f"  outputs/{source}   ->  make exp EXP={source.split('/')[0]}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
