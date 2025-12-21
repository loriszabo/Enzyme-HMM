from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


DEFAULT_PER_POSITION = "per_position_mean_occupancy_and_mutation_densities.csv"
DEFAULT_CURVES = "curves_roc_pr.csv"
DEFAULT_SYNTHETIC = "synthetic_aid_data_2.csv"


def _set_pub_style(*, font: str = "DejaVu Sans", base_size: float = 10.0) -> None:
    """
    Conservative, publication-ready Matplotlib style that works out-of-the-box.
    Produces good PDF output and readable PNGs.
    """
    mpl.rcParams.update(
        {
            # Typography
            "font.family": "sans-serif",
            "font.sans-serif": [font, "Arial", "Helvetica", "Liberation Sans"],
            "font.size": base_size,
            "axes.titlesize": base_size + 1,
            "axes.labelsize": base_size,
            "xtick.labelsize": base_size - 1,
            "ytick.labelsize": base_size - 1,
            "legend.fontsize": base_size - 1,
            # Lines
            "lines.linewidth": 2.0,
            "lines.markersize": 4.0,
            # Axes
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
            # Figure
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            # PDF/SVG text handling
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    if window is None or int(window) <= 1:
        return y
    s = pd.Series(y)
    return s.rolling(window=int(window), center=True, min_periods=1).mean().to_numpy(dtype=float)


def _auc_trapz(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 2:
        return float("nan")
    xs = x[m]
    ys = y[m]
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    return float(np.trapz(ys, xs))


def _save(fig: plt.Figure, outpath: Path, *, fmt: str = "pdf", dpi: int = 300) -> None:
    outpath = outpath.with_suffix(f".{fmt.lower()}")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=dpi)
    print(f"Saved: {outpath}")


def _parse_01_sequence(val: object, *, expected_len: Optional[int] = None) -> Optional[np.ndarray]:
    """
    Parse a per-position 0/1 sequence from CSV cell.
    Supports:
      - bitstring like "010011"
      - comma/space separated like "0,1,0,0,1,1" or "0 1 0 0 1 1"
    Returns float array of shape (T,) with values in {0.0, 1.0}, or None on failure.
    """
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return None

    s = str(val).strip()
    if not s:
        return None

    # Case 1: pure bitstring
    if set(s).issubset({"0", "1"}):
        arr = np.fromiter((1.0 if c == "1" else 0.0 for c in s), dtype=float, count=len(s))
        if expected_len is not None and len(arr) != int(expected_len):
            return None
        return arr

    # Case 2: tokenized list
    s2 = s.replace(",", " ").replace(";", " ").replace("\t", " ")
    toks = [t for t in s2.split(" ") if t != ""]
    if not toks:
        return None
    try:
        arr_i = np.array([int(t) for t in toks], dtype=int)
    except ValueError:
        return None

    if not np.all((arr_i == 0) | (arr_i == 1)):
        return None

    if expected_len is not None and len(arr_i) != int(expected_len):
        return None

    return arr_i.astype(float)


def _per_position_means_from_synthetic(
    synthetic_csv: Path,
    *,
    max_len: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Reads synthetic CSV and computes per-position means for:
      - hidden_states (true state)
      - fire_events (emission indicator)
    Returns (mean_hidden, mean_fire), each shape (max_len,) padded with NaN where undefined.
    """
    if not synthetic_csv.exists():
        return None, None

    df = pd.read_csv(synthetic_csv)

    mean_hidden = None
    mean_fire = None

    if "hidden_states" in df.columns:
        hs_mat = np.full((len(df), max_len), np.nan, dtype=float)
        for i in range(len(df)):
            hs = _parse_01_sequence(df.loc[df.index[i], "hidden_states"])
            if hs is None:
                continue
            T = min(len(hs), max_len)
            hs_mat[i, :T] = hs[:T]
        mean_hidden = np.nanmean(hs_mat, axis=0)

    if "fire_events" in df.columns:
        fe_mat = np.full((len(df), max_len), np.nan, dtype=float)
        for i in range(len(df)):
            fe = _parse_01_sequence(df.loc[df.index[i], "fire_events"])
            if fe is None:
                continue
            T = min(len(fe), max_len)
            fe_mat[i, :T] = fe[:T]
        mean_fire = np.nanmean(fe_mat, axis=0)

    return mean_hidden, mean_fire


def plot_per_position(
    per_pos_csv: Path,
    *,
    outdir: Path,
    fmt: str,
    dpi: int,
    smooth_window: int = 1,
    xlim: Optional[Tuple[int, int]] = None,
    synthetic_csv: Optional[Path] = None,
) -> None:
    df = pd.read_csv(per_pos_csv)
    required = {"pos", "mean_posterior_occupancy", "mutation_rate_any", "mutation_rate_CtoT"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{per_pos_csv} missing required columns: {sorted(missing)}")

    x = df["pos"].to_numpy(dtype=int)
    occ = df["mean_posterior_occupancy"].to_numpy(dtype=float)
    m_any = df["mutation_rate_any"].to_numpy(dtype=float)
    m_ct = df["mutation_rate_CtoT"].to_numpy(dtype=float)

    occ_s = _rolling_mean(occ, smooth_window)
    m_any_s = _rolling_mean(m_any, smooth_window)
    m_ct_s = _rolling_mean(m_ct, smooth_window)

    # Any-other-than-C->T = (any) - (C->T). Clamp for numerical/smoothing safety.
    m_other_s = np.maximum(0.0, m_any_s - m_ct_s)

    # Optional: overlay truth + emissions from synthetic CSV
    mean_hidden_s = None
    mean_fire_s = None
    if synthetic_csv is not None:
        mh, mf = _per_position_means_from_synthetic(synthetic_csv, max_len=len(x))
        if mh is not None:
            mean_hidden_s = _rolling_mean(mh[: len(x)], smooth_window)
        if mf is not None:
            mean_fire_s = _rolling_mean(mf[: len(x)], smooth_window)

    # Colors
    c_occ = "#2C7FB8"    # blue
    c_ct = "#D95F0E"     # orange (C→T, upwards)
    c_other = "#6A51A3"  # purple (other, downwards)
    c_truth = "#222222"  # dark gray/black
    c_fire = "#1B9E77"   # green

    # ---- Figure 1: occupancy (top) + diverging mutation rates (bottom)
    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(6.6, 4.6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.25], "hspace": 0.08},
    )

    # Top: (optional) true hidden mean + (optional) fire_events mean
    # Plot these first (lower z-order), then posterior on top with transparency.
    if mean_hidden_s is not None and np.any(np.isfinite(mean_hidden_s)):
        ax_top.plot(
            x,
            mean_hidden_s,
            color=c_truth,
            linestyle="--",
            linewidth=1.8,
            label="True hidden state mean  E[S_t] (synthetic)",
            zorder=3,
        )

    if mean_fire_s is not None and np.any(np.isfinite(mean_fire_s)):
        ax_top.plot(
            x,
            mean_fire_s,
            color=c_fire,
            linestyle=":",
            linewidth=2.0,
            label="Mean fire_events  E[fire_t] (synthetic)",
            zorder=3,
        )

    # Posterior occupancy LAST so it is the top layer; slightly transparent
    ax_top.plot(
        x,
        occ_s,
        color=c_occ,
        alpha=0.75,
        linewidth=2.4,
        label="Posterior occupancy  P(state=1|data)",
        zorder=5,
    )

    ax_top.set_ylabel("Mean (0–1)")
    ax_top.set_ylim(0.0, 1.0)
    ax_top.legend(loc="upper right", frameon=False)
    ax_top.set_title("Per-position posterior occupancy, true state mean, and emissions")


    # Bottom: diverging mutation plot around y=0
    bar_w = 1.0
    ax_bot.axhline(0.0, color="black", linewidth=1.0, alpha=0.7, zorder=2)

    ax_bot.bar(
        x,
        m_ct_s,
        width=bar_w,
        color=c_ct,
        alpha=0.85,
        linewidth=0.0,
        label="C→T mutation rate (up)",
        zorder=3,
    )
    ax_bot.bar(
        x,
        -m_other_s,
        width=bar_w,
        color=c_other,
        alpha=0.75,
        linewidth=0.0,
        label="Other mutation rate (down) = any − C→T",
        zorder=3,
    )

    max_mut = float(np.nanmax(np.r_[m_ct_s, m_other_s])) if np.isfinite(np.nanmax(np.r_[m_ct_s, m_other_s])) else 0.0
    ylim = 1.10 * max_mut if max_mut > 0 else 1.0
    ax_bot.set_ylim(-ylim, ylim)

    ax_bot.set_xlabel("Position")
    ax_bot.set_ylabel("Mean mutation rate\n(± around 0)")
    ax_bot.legend(loc="upper right", frameon=False)

    ax_bot.grid(True, axis="y")
    ax_bot.grid(False, axis="x")

    if xlim is not None:
        ax_bot.set_xlim(xlim)

    _save(fig, outdir / "fig_per_position_occupancy_truth_emissions_mutation", fmt=fmt, dpi=dpi)
    plt.close(fig)

    # ---- Figure 2: single-axis plot should be ONLY C→T mutation rate
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ax.plot(x, m_ct_s, color=c_ct, label="Mutation rate (C→T)")
    ax.set_xlabel("Position")
    ax.set_ylabel("Mean C→T mutation rate")
    ax.set_ylim(bottom=0.0)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("Per-position C→T mutation rate")
    _save(fig, outdir / "fig_per_position_ct_mutation_rate", fmt=fmt, dpi=dpi)
    plt.close(fig)


def plot_curves(
    curves_csv: Path,
    *,
    outdir: Path,
    fmt: str,
    dpi: int,
) -> None:
    df = pd.read_csv(curves_csv)
    if "curve" not in df.columns:
        raise ValueError(f"{curves_csv} must contain a 'curve' column with values 'ROC'/'PR'.")

    # ---- ROC
    roc = df[df["curve"] == "ROC"].copy()
    if not roc.empty:
        if not {"fpr", "tpr"}.issubset(roc.columns):
            raise ValueError(f"{curves_csv}: ROC rows must contain columns 'fpr' and 'tpr'.")
        fpr = roc["fpr"].to_numpy(dtype=float)
        tpr = roc["tpr"].to_numpy(dtype=float)
        auc = _auc_trapz(fpr, tpr)

        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        ax.plot(fpr, tpr, color="#2C7FB8", label=f"ROC (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], color="black", alpha=0.4, linewidth=1.2, linestyle="--", label="Chance")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="lower right", frameon=False)
        ax.set_title("ROC curve")
        _save(fig, outdir / "fig_roc_curve", fmt=fmt, dpi=dpi)
        plt.close(fig)
    else:
        print(f"Warning: no ROC rows found in {curves_csv}")

    # ---- PR
    pr = df[df["curve"] == "PR"].copy()
    if not pr.empty:
        if not {"recall", "precision"}.issubset(pr.columns):
            raise ValueError(f"{curves_csv}: PR rows must contain columns 'recall' and 'precision'.")
        recall = pr["recall"].to_numpy(dtype=float)
        precision = pr["precision"].to_numpy(dtype=float)
        auc = _auc_trapz(recall, precision)

        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        ax.plot(recall, precision, color="#D95F0E", label=f"PR (AUC={auc:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="lower left", frameon=False)
        ax.set_title("Precision–Recall curve")
        _save(fig, outdir / "fig_pr_curve", fmt=fmt, dpi=dpi)
        plt.close(fig)
    else:
        print(f"Warning: no PR rows found in {curves_csv}")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot posterior_state_probabilities.py outputs (publication-ready).")
    p.add_argument("--per-position", type=str, default=DEFAULT_PER_POSITION, help="Per-position CSV output path.")
    p.add_argument("--curves", type=str, default=DEFAULT_CURVES, help="ROC/PR curves CSV output path.")
    p.add_argument("--synthetic-csv", type=str, default=DEFAULT_SYNTHETIC, help="Synthetic CSV with hidden_states and fire_events.")
    p.add_argument("--outdir", type=str, default="figures_posterior_outputs", help="Directory to write figures.")
    p.add_argument("--fmt", type=str, default="pdf", choices=["pdf", "png", "svg"], help="Output format.")
    p.add_argument("--dpi", type=int, default=300, help="DPI for raster outputs (PNG).")
    p.add_argument("--smooth-window", type=int, default=1, help="Centered rolling mean window (>=1).")
    p.add_argument("--font", type=str, default="DejaVu Sans", help="Preferred sans-serif font.")
    p.add_argument("--base-fontsize", type=float, default=10.0, help="Base font size in points.")
    p.add_argument("--xlim", type=str, default=None, help="Optional x-limits as 'start,end' (inclusive).")

    args = p.parse_args()

    _set_pub_style(font=args.font, base_size=float(args.base_fontsize))

    per_pos_path = Path(args.per_position)
    curves_path = Path(args.curves)
    synthetic_path = Path(args.synthetic_csv) if args.synthetic_csv else None
    outdir = Path(args.outdir)

    xlim = None
    if args.xlim:
        a, b = args.xlim.split(",")
        xlim = (int(a), int(b))

    if per_pos_path.exists():
        plot_per_position(
            per_pos_path,
            outdir=outdir,
            fmt=args.fmt,
            dpi=int(args.dpi),
            smooth_window=int(args.smooth_window),
            xlim=xlim,
            synthetic_csv=synthetic_path,
        )
    else:
        print(f"Warning: per-position CSV not found: {per_pos_path}")

    if curves_path.exists():
        plot_curves(curves_path, outdir=outdir, fmt=args.fmt, dpi=args.dpi)
    else:
        print(f"Warning: curves CSV not found: {curves_path}")


if __name__ == "__main__":
    main()