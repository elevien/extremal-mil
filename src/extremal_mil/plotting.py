"""Shared plotting for the trajectory figures (``notebooks/fig2.ipynb``).

The trajectory figures are all the same object drawn from different data: a
grid with one column per :math:`d` and three rows -- the loss against flow
time, then the :math:`(u,\\lambda)` and :math:`(\\rho,\\lambda)` projections of
the same runs. :func:`new_grid`, :func:`draw_runs`, :func:`draw_boundaries`
and :func:`finish_grid` are the four steps of building one, so a notebook
cell that draws GD and a cell that draws the gradient flow differ only in
what they pass to :func:`draw_runs`.

Nothing here computes anything: trajectories come from
``fig2_experiment.load_trajectory_sweep`` or from integrating
``ode.gradient_flow_rhs``, and the phase boundaries come from ``theory``.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from . import theory

# Shared look for the two phase boundaries, so every figure draws them alike.
HL_STYLE = dict(color="0.45", ls="-", lw=1.2)
RN_STYLE = dict(color="C1", ls="-", lw=1.3)
# Washes marking the two regions a boundary separates: gray below
# lambda = lambda* (high temperature), orange right of the retrieval
# boundary. They overlap in HR, which reads as both.
H_SHADE = dict(color="0.5", alpha=0.10, lw=0, zorder=0)
R_SHADE = dict(color="C1", alpha=0.09, lw=0, zorder=0)
GD_BACKDROP_STYLE = dict(color="0.78", lw=1.0)
# The asymptotic field, wherever it is drawn against finite-size measurements.
THEORY_STYLE = dict(color="0.15", ls="--", lw=1.3)
# Phase names written into the regions themselves rather than a legend.
# The regions are full of trajectories, so the names carry a background box
# rather than being placed in whatever corner happens to be empty.
PHASE_LABEL_STYLE = dict(color="0.25", fontsize=8, ha="center", va="center",
                         zorder=6, fontweight="light",
                         bbox=dict(facecolor="white",
                                   edgecolor="none", boxstyle="round,pad=0.20"))

# Viridis stopped short of its yellow end, which sits at only 1.3:1 contrast
# against white and disappears in print. Truncating at 0.80 raises that to
# 1.9:1 while staying perceptually uniform, and keeps the top of the scale
# green rather than orange, so it is not confused with the retrieval
# boundary and its wash.
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "viridis_dark", plt.get_cmap("viridis")(np.linspace(0.0, 0.80, 256)))

# Every paper figure is drawn at the width it is printed at (revtex: one
# column is 246 pt, the full text 510 pt), so these sizes are the sizes on the
# page: 8 pt text, ticks and panel titles, 9 pt axis labels and figure titles.
COL_W = 3.4
TEXT_W = 7.0

RCPARAMS = {
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 8,
    "figure.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.2,
    "legend.handlelength": 1.6,
    "savefig.pad_inches": 0.02,
}


def run_colors(x0, cmap=None):
    """Per-run colors for an initial-condition grid, plus the colorbar mappable.

    Returns ``(colors, mappable)``: one color per entry of ``x0``, and the
    ``ScalarMappable`` to hand to :func:`finish_grid` for the colorbar.
    """
    cmap = cmap or SEQUENTIAL_CMAP
    norm = plt.Normalize(np.min(x0), np.max(x0))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return [cmap(norm(v)) for v in x0], sm




def new_row(n_cols, width=TEXT_W, height=2.3, fig=None):
    """Empty 1 x ``n_cols`` row sharing an x-axis, for curves against one variable.

    ``fig`` may be an existing (sub)figure to fill instead of a new one.
    """
    if fig is not None:
        return fig, fig.subplots(1, n_cols, sharex=True, squeeze=False)
    return plt.subplots(1, n_cols, figsize=(width, height),
                        sharex=True, squeeze=False, layout="constrained")


def rn_threshold(ax, u_star):
    """R/N boundary where it is a single point in ``u`` (fixed lambda): a rule at
    ``u_star``, drawn in the style :func:`rn_boundary` uses for the curve."""
    ax.axvline(u_star, zorder=1, **RN_STYLE)


def zero_line(ax):
    """Mark zero, which for a derivative panel separates growth from decay."""
    ax.axhline(0.0, color="0.75", lw=0.8, zorder=0)


def hl_boundary(ax, alpha, shade=True):
    """High/low-temperature boundary :math:`\\lambda=\\lambda^*`, flat in every
    plane, with the high-temperature side below it shaded.

    Reads the axis limits, so call it once they are final: the shading then
    runs to the edges of the panel rather than to the extent of whatever
    data happens to be plotted.
    """
    lam_star = theory.lambda_star(alpha)
    y_lo, y_hi = ax.get_ylim()
    if shade:
        ax.axhspan(y_lo, min(lam_star, y_hi), **H_SHADE)
    ax.axhline(lam_star, zorder=1, **HL_STYLE)
    ax.set_ylim(y_lo, y_hi)


def rn_boundary(ax, alpha, rho_true, shade=True):
    """Retrieval/non-retrieval boundary in the :math:`(u,\\lambda)` plane.

    The criterion is :math:`\\rho_{\\rm true}u=\\phi_\\alpha(\\lambda)`
    (``theory.phi_alpha``). ``phi_alpha`` diverges as :math:`\\lambda\\to0`, so
    the grid is floored at ``1e-3``; that floor moves the drawn curve only,
    never a trajectory.

    Reads the axis limits, so call it once they are final: the curve and its
    shading then span the panel rather than the range of the plotted data.
    """
    y_lo, y_hi = ax.get_ylim()
    x_lo, x_hi = ax.get_xlim()
    lam_grid = np.linspace(max(y_lo, 1e-3), y_hi, 200)
    u_boundary = np.array([theory.phi_alpha(l, alpha) for l in lam_grid]) / rho_true
    if shade:
        ax.fill_betweenx(lam_grid, u_boundary, x_hi, **R_SHADE)
    ax.plot(u_boundary, lam_grid, zorder=2, **RN_STYLE)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(x_lo, x_hi)


def phase_labels(ax, alpha, rho_true, margin=0.05, bottom_margin=None):
    """Write HR/LR/HN/LN into the four outer corners of a :math:`(u,\\lambda)`
    panel: LN/HN against the left edge, LR/HR against the right, split
    top/bottom at :math:`\\lambda^\\star`.

    Call after the axis limits are final. ``margin`` (a fraction of each
    axis's range) insets every label from the plot's own border, so the
    text never overlaps it; the same margin also keeps the label clear of
    the retrieval boundary behind it, falling back to just inside that
    boundary rather than the true corner if a region is too thin there for
    both margins to hold at once.
    """
    # Axes coordinates, anchored by the label's own outer corner, so the label
    # box sits inside the spines on linear and log axes alike.
    style = {**PHASE_LABEL_STYLE, "fontsize": 7}
    bottom = margin if bottom_margin is None else bottom_margin
    for name, x, y, ha, va in (
        ("LN", margin, 1 - margin, "left", "top"),
        ("LR", 1 - margin, 1 - margin, "right", "top"),
        ("HN", margin, bottom, "left", "bottom"),
        ("HR", 1 - margin, bottom, "right", "bottom"),
    ):
        ax.text(x, y, name, transform=ax.transAxes,
                **{**style, "ha": ha, "va": va})


def temperature_labels(ax, alpha, pad=0.45):
    """Write H/L either side of :math:`\\lambda=\\lambda^*` on a panel where the
    retrieval boundary cannot be drawn, such as :math:`(\\rho,\\lambda)`."""
    lam_star = theory.lambda_star(alpha)
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    x = x_lo + (x_hi - x_lo) * 0.08
    for name, a, b in (("L", lam_star, y_hi), ("H", lam_star, y_lo)):
        ax.text(x, a + (b - a) * (0.5 + 0.5 * pad), name, **PHASE_LABEL_STYLE)


def draw_boundaries(col_axes, alpha, rho_true):
    """Both boundaries on one column of :func:`new_grid`.

    R/N depends only on :math:`(u,\\lambda)`, so it has no rho-dependence to
    draw in the second row. Call after the runs are drawn and the limits are
    final, so both regions reach the edges of their panels; the styles carry
    low ``zorder``, so they still render underneath the trajectories.
    """
    for ax in col_axes[:2]:
        hl_boundary(ax, alpha)
    rn_boundary(col_axes[0], alpha, rho_true)


def draw_runs(col_axes, t, lam, u, rho, loss, colors, ends=True,plot_loss=True, end_marker="o", **kw):
    """One set of runs on one column: the two projections, then loss vs ``t``.

    ``lam``/``u``/``rho``/``loss`` are ``(n_runs, n_points)``; ``t`` is
    ``(n_points,)``. ``ends=True`` marks each run's start with an open circle
    and its end with a filled one. Extra kwargs go to ``plot``.
    """
    for i, c in enumerate(colors):
        col_axes[0].plot(u[i], lam[i], color=c, **kw)
        col_axes[1].plot(rho[i], lam[i], color=c, **kw)
        if plot_loss:
            col_axes[2].plot(t, loss[i], color=c, **kw)
        if ends:
            for ax, x in ((col_axes[0], u), (col_axes[1], rho)):
                ax.plot(x[i][0], lam[i][0],end_marker, ms=3, mfc="white", mec=c,
                        mew=1.1, zorder=4)
                ax.plot(x[i][-1], lam[i][-1],end_marker, ms=3.5, color=c, mec="white",
                        mew=0.6, zorder=5)


def draw_backdrop(col_axes, t, lam, u, rho, loss):
    """Draw runs in grey behind everything, as context rather than subject."""
    for i in range(len(lam)):
        col_axes[0].plot(u[i], lam[i], zorder=2, **GD_BACKDROP_STYLE)
        col_axes[1].plot(rho[i], lam[i], zorder=2, **GD_BACKDROP_STYLE)
        col_axes[2].plot(t, loss[i], zorder=2, **GD_BACKDROP_STYLE)


def finish_grid(fig, axes, mappable, t_max, loss_lim, suptitle, cbar_label,
                handles=None, u_lim=(-1.1, 1.1), rho_lim=(1.0, 2.85),plot_loss=True):
    """Axis labels, limits, colorbar, legend and title for a :func:`new_grid`.

    ``loss_lim`` is passed explicitly because the gradient flow reaches zero
    asymptotic loss on the retrieval branch; left to autoscale, its curve
    stretches the log axis by ~25 decades and flattens everything else.
    """
    if plot_loss:
        axes[2, 0].set_yscale("log")
        axes[2, 0].set_ylim(*loss_lim)
        axes[2, 0].set_xlim(0, t_max)
    axes[0, 0].set_xlim(*u_lim)
    axes[1, 0].set_xlim(*rho_lim)
    for col in range(axes.shape[1]):
        axes[0, col].set_xlabel("$u$")
        axes[1, col].set_xlabel(r"$\rho$")
        if plot_loss:
            axes[2, col].set_xlabel(r"$t = \mathrm{lr}\cdot\mathrm{step}$")
    if plot_loss:
        axes[2, 0].set_ylabel(r"$\bar{L}$")
    for row in (0, 1):
        axes[row, 0].set_ylabel(r"$\lambda$")

    cbar = fig.colorbar(mappable, ax=axes, fraction=0.022, pad=0.012)
    cbar.set_label(cbar_label)
    cbar.outline.set_visible(False)
    if handles:
        fig.legend(handles=handles, loc="outside lower center",
                   ncols=len(handles), frameon=False)
    if suptitle:
        fig.suptitle(suptitle)


def theory_handle(label="gradient flow (theory)"):
    """Legend handle for a curve drawn in :data:`THEORY_STYLE`."""
    return Line2D([], [], label=label, **THEORY_STYLE)


def series_handles(values, colors, fmt):
    """Legend handles for one colored curve per swept value, e.g. per ``d``."""
    return [Line2D([], [], color=c, lw=1.3, label=fmt(v))
            for v, c in zip(values, colors)]


def finish_row(fig, axes, suptitle, xlabel, handles=None):
    """Shared x-label, legend and title for a :func:`new_row` figure."""
    for ax in axes[0]:
        ax.set_xlabel(xlabel)
    if handles:
        # two rows past four entries, so the legend never outgrows the page
        ncols = len(handles) if len(handles) <= 4 else -(-len(handles) // 2)
        fig.legend(handles=handles, loc="outside lower center",
                   ncols=ncols, frameon=False)
    if suptitle:
        fig.suptitle(suptitle)


def rn_handle(label="R/N boundary"):
    """Legend handle for the retrieval boundary, however it was drawn."""
    return Line2D([], [], label=label, **RN_STYLE)


def boundary_handles():
    """Legend handles for the two phase boundaries."""
    return [Line2D([], [], label="H/L boundary", **HL_STYLE), rn_handle()]


def endpoint_handles(marker="o"):
    """Legend handles for the start/end markers drawn by :func:`draw_runs`;
    pass the ``end_marker`` given there."""
    return [Line2D([], [], color="0.35", marker=marker, ms=3, mfc="white",
                   ls="none", label="start"),
            Line2D([], [], color="0.35", marker=marker, ms=3.5, ls="none",
                   label="end")]
