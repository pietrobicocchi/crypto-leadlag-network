"""The project's visual language.

Every figure goes through this module. Change a default here and every figure
changes with it. Method colours and labels are semantic: the estimator and the
baseline it replaces keep the same colour and the same name in every figure of
the report.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- semantic constants ------------------------------------------------------
# One entry per method. Never hard-code a colour or a display name elsewhere.

METHOD_LABELS = {
    "hayashi_yoshida": "Hayashi–Yoshida",
    "gridded": "Gridded baseline",
}

METHOD_COLORS = {
    "hayashi_yoshida": "#0B6E4F",
    "gridded": "#8A8A8A",  # grey: the method that fails, in every figure it fails in
}

# --- page geometry, in inches ------------------------------------------------

COLUMN_WIDTH = 3.4
PAGE_WIDTH = 7.0


def setup_style() -> None:
    """Apply the project's matplotlib defaults. Called on import."""
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,  # editable text in the PDF, not outlines
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.4,
            "lines.markersize": 4,
            "axes.prop_cycle": mpl.cycler(color=list(METHOD_COLORS.values())),
        }
    )


def new_figure(width: float = COLUMN_WIDTH, height: float | None = None, **kwargs):
    """Return `(fig, ax)` sized for the page. `kwargs` go to `plt.subplots`."""
    if height is None:
        height = width * 0.66
    return plt.subplots(figsize=(width, height), **kwargs)


def save_figure(fig, path: str | Path) -> Path:
    """Write `fig` to `path` (PDF preferred) and close it."""
    path = Path(path)
    if path.suffix.lower() not in {".pdf", ".svg"}:
        raise ValueError(f"report figures must be vector (.pdf/.svg), got {path.suffix!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def label_panel(ax, label: str) -> None:
    """Put a panel letter ("a", "b", ...) at the top-left of `ax`."""
    ax.text(
        -0.15,
        1.05,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


setup_style()
