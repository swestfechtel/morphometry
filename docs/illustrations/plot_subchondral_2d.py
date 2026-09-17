#!/usr/bin/env python3
"""Render the subchondral-distance ray tracing as three orthogonal 2D slices.

Companion to ``plot_subchondral_3d.py``. Loads a NaKo hip segmentation (isotropic;
femur=1, cartilage=2, acetabulum=3) together with its PD MRI, splits both into the
requested image-side half, runs the *actual* measurement
``morphometry.measurements.hip.calculate_subchondral_distance_ray_tracing`` with the
library's default cone axis, and draws sagittal, coronal and axial slices through the
femoral head centre (FHC) the measurement itself uses.

Each panel shows the MRI with the femur, full pelvis label and peri-acetabular crop
outlined, the FHC and the femoral-head circle, and the rays. Rays are 3D, so every
panel draws two things, deliberately kept visually distinct:

* **bold** — rays whose head exit *and* acetabulum hit both lie within ``--slab`` voxels
  of the displayed slice, i.e. rays that genuinely run in that plane;
* **faint** — the projection of *all* surviving rays onto the plane.

With the default ``'superior'`` axis the axial slice through the FHC is perpendicular to
the cone axis, so no ray lies in it: the axial panel shows only the projected footprint
of the sampled acetabular roof, and marks the cone axis as pointing out of the page.

    MORPH_NAKO_SAMPLE_DIR=/path/to/segmentations \\
        python docs/illustrations/plot_subchondral_2d.py [--case 0] [--side left] [--slab 1.5]

The MRI is looked up as ``<nifti dir>/<segmentation stem>_0000.nii.gz`` (override the
directory with ``MORPH_NAKO_IMAGE_DIR``); without it the panels fall back to the
segmentation alone. This is a didactic render, not a clinical figure.
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent.parent))  # repo root
sys.path.append(str(HERE))                # the 3D companion script

from morphometry.hip import get_femoral_head_center  # noqa: E402
from morphometry.image_io import Image, Segmentation, split_image_left_right  # noqa: E402
from morphometry.measurements import hip as H  # noqa: E402
from plot_subchondral_3d import (DEFAULT_AXIS, _cone_axis,  # noqa: E402
                                 _peri_acetabular_mask, _sample_files)

DEFAULT_IMAGE_DIR = Path(os.environ.get(
    "MORPH_NAKO_IMAGE_DIR", "/home/simon/Data/NaKo_sample/nifti"))

# colours shared with the 3D renders so the two figures read as one set
C_RAY, C_EXIT, C_HIT = "#DC2626", "#2563EB", "#0E9F6E"
C_FEMUR, C_PELVIS, C_CROP, C_AXIS = "#F5E6B8", "#C9B37E", "#E0A526", "#22D3EE"

# (panel title, index axis held fixed, in-plane (horizontal, vertical) index axes,
#  edge labels (left, right, top, bottom)). Arrays are LPI: index 0 runs toward patient
#  Left, 1 toward Posterior, 2 toward Inferior — so with row index increasing downward
#  every panel comes out in radiological orientation (patient right on the viewer's
#  left, anterior up in the axial view, superior up in the other two).
VIEWS = (
    ("Sagittal", 0, (1, 2), ("A", "P", "S", "I")),
    ("Coronal", 1, (0, 2), ("R", "L", "S", "I")),
    ("Axial", 2, (0, 1), ("R", "L", "A", "P")),
)


def _load_pair(index: int, side: str) -> tuple[Segmentation, np.ndarray | None, str]:
    """Load the ``index``-th segmentation and its MRI, both LPI and split to ``side``.

    The segmentation goes through the same ``transform_coordinate_system`` +
    ``remove_outliers`` + half-split as the pipelines and the 3D script; the MRI gets
    the identical reorientation and split (no outlier removal — it is an intensity
    image), so the two stay voxel-aligned.
    :param index: Index into the sorted NaKo segmentation files.
    :param side: 'left' or 'right' (image side).
    :return: ``(side_segmentation, side_mri_array_or_None, case_id)``.
    """
    path = _sample_files()[index % len(_sample_files())]
    seg = Segmentation("nibabel")
    seg.read_image(str(path))
    seg.transform_coordinate_system()
    seg.remove_outliers()
    half = seg.array.shape[0] // 2
    side_seg = split_image_left_right(seg)[0 if side == "left" else 1]

    mri = None
    img_path = DEFAULT_IMAGE_DIR / path.name.replace(".nii.gz", "_0000.nii.gz")
    if img_path.exists():
        img = Image("nibabel")
        img.read_image(str(img_path))
        img.transform_coordinate_system()
        if img.array.shape == seg.array.shape:
            mri = img.array[:half] if side == "left" else img.array[half:]
        else:
            print(f"warning: MRI shape {img.array.shape} != segmentation "
                  f"{seg.array.shape}; drawing without the MRI underlay")
    else:
        print(f"warning: no MRI at {img_path}; drawing the segmentation only")
    return side_seg, mri, path.name.split("_")[0]


def _to_index(image: Segmentation, points_phys: np.ndarray) -> np.ndarray:
    """Map (N, 3) physical points to continuous (N, 3) index coordinates."""
    affine = np.asarray(image.affine)
    linear, offset = affine[:3, :3], affine[:3, 3]
    return (np.atleast_2d(points_phys) - offset) @ np.linalg.inv(linear).T


def _slice(volume: np.ndarray, fixed_axis: int, position: int) -> np.ndarray:
    """The 2D slice at ``position`` along ``fixed_axis``, transposed to (row=vertical, col=horizontal)."""
    return np.take(volume, position, axis=fixed_axis).T


def _draw_panel(ax: plt.Axes, view: tuple, seg: Segmentation, mri: np.ndarray | None,
                crop: np.ndarray, fhc_idx: np.ndarray, r_idx: float, axis_phys: np.ndarray,
                cone_angle: float, exits_idx: np.ndarray, hits_idx: np.ndarray,
                slab: float, half_width: float) -> int:
    """Draw one orthogonal slice through the FHC and return how many rays lie in its slab.

    :param ax: The matplotlib Axes to draw into.
    :param view: One entry of ``VIEWS``.
    :param seg: The side segmentation (LPI).
    :param mri: The side MRI array, voxel-aligned with ``seg``, or None.
    :param crop: Boolean peri-acetabular crop mask, same grid as ``seg``.
    :param fhc_idx: The femoral head centre in index coordinates.
    :param r_idx: The femoral head radius in index units.
    :param axis_phys: The unit cone axis in physical space.
    :param cone_angle: The cone half-angle in degrees.
    :param exits_idx: (N, 3) head-exit points in index coordinates.
    :param hits_idx: (N, 3) acetabulum-hit points in index coordinates.
    :param slab: Half-thickness, in voxels, of the slab whose rays count as in-plane.
    :param half_width: Half-width of the displayed window around the FHC, in voxels.
    :return: The number of rays drawn as in-plane.
    """
    title, fixed, (hu, hv), (left, right, top, bottom) = view
    pos = int(round(fhc_idx[fixed]))
    ax.set_facecolor("black")  # the window can run past the volume edge
    arr = seg.array

    if mri is not None:
        sl = _slice(mri, fixed, pos)
        lo, hi = np.percentile(sl[sl > 0], [1, 99.5]) if np.any(sl > 0) else (0, 1)
        ax.imshow(sl, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    else:
        ax.imshow(np.zeros(_slice(arr, fixed, pos).shape), cmap="gray", vmin=0, vmax=1)

    # label outlines: femur and the crop that restricts the hit search, full pelvis faint
    for mask, colour, lw, alpha in ((arr == 3, C_PELVIS, 0.8, 0.55),
                                    (crop, C_CROP, 1.6, 0.95),
                                    (arr == 1, C_FEMUR, 1.6, 0.95)):
        m2 = _slice(mask.astype(float), fixed, pos)
        if m2.any():
            ax.contour(m2, levels=[0.5], colors=colour, linewidths=lw, alpha=alpha)

    # all rays projected (faint), then the rays genuinely running in this slice (bold)
    for e, h in zip(exits_idx, hits_idx):
        ax.plot([e[hu], h[hu]], [e[hv], h[hv]], color=C_RAY, lw=0.6, alpha=0.35)
    in_slab = (np.abs(exits_idx[:, fixed] - pos) <= slab) & (np.abs(hits_idx[:, fixed] - pos) <= slab)
    for e, h in zip(exits_idx[in_slab], hits_idx[in_slab]):
        ax.plot([e[hu], h[hu]], [e[hv], h[hv]], color=C_RAY, lw=1.8)
    ax.scatter(exits_idx[in_slab][:, hu], exits_idx[in_slab][:, hv], s=14, c=C_EXIT, zorder=4)
    ax.scatter(hits_idx[in_slab][:, hu], hits_idx[in_slab][:, hv], s=14, c=C_HIT, zorder=4)

    # femoral head: the fitted sphere's great circle in this plane, and its centre
    ax.add_patch(Circle((fhc_idx[hu], fhc_idx[hv]), r_idx, fill=False, ls="--",
                        lw=1.2, ec="white", alpha=0.85))
    ax.plot(fhc_idx[hu], fhc_idx[hv], marker="+", ms=14, mew=2.2, color="white", zorder=5)

    # cone axis and cone edges. Angles are measured in millimetres (index offsets scaled by
    # the in-plane voxel spacing) so they stay true on anisotropic grids, then mapped back
    # to index coordinates to register with the image and the rays.
    scale = np.linalg.norm(np.asarray(seg.affine)[:3, :3], axis=0)[[hu, hv]]
    axis_idx = _to_index(seg, np.asarray(seg.transform_index_to_physical_point(fhc_idx)) +
                         axis_phys)[0] - fhc_idx
    in_plane = np.array([axis_idx[hu], axis_idx[hv]])
    # kept inside the window: an annotation whose tip falls outside the axes is dropped
    # entirely by matplotlib rather than clipped
    length = 0.92 * half_width
    edge_text = ""
    if np.linalg.norm(in_plane) > 0.25 * np.linalg.norm(axis_idx):
        d_mm = in_plane * scale
        d_mm /= np.linalg.norm(d_mm)

        def _to_idx_point(direction_mm: np.ndarray) -> tuple[float, float]:
            """Point ``length`` along a millimetre-space direction from the FHC, in index coords."""
            step = direction_mm / scale
            step *= length / np.linalg.norm(step)
            return fhc_idx[hu] + step[0], fhc_idx[hv] + step[1]

        # actual cone edges: the angular extent of the rays cast in this slice, measured
        # from the axis to each in-slab ray (FHC, exit and hit are collinear), rather than
        # the nominal half-angle - lateral rim overshoot and the slab make them differ
        vec = (hits_idx[in_slab][:, [hu, hv]] - fhc_idx[[hu, hv]]) * scale
        if len(vec) >= 2:
            ang = np.degrees(np.arctan2(d_mm[0] * vec[:, 1] - d_mm[1] * vec[:, 0], vec @ d_mm))
            for a in (ang.min(), ang.max()):
                t = np.radians(a)
                e = np.array([d_mm[0] * np.cos(t) - d_mm[1] * np.sin(t),
                              d_mm[0] * np.sin(t) + d_mm[1] * np.cos(t)])
                ex, ey = _to_idx_point(e)
                ax.plot([fhc_idx[hu], ex], [fhc_idx[hv], ey], color=C_AXIS, lw=1.4, ls="--", alpha=0.95)
                ax.text(ex, ey, f"{abs(a):.0f}°", color=C_AXIS, fontsize=9, fontweight="bold",
                        ha="center", va="bottom")
            edge_text = f"; cone edges {abs(ang.min()):.0f}° / {abs(ang.max()):.0f}° (nominal ±{cone_angle:g}°)"

        tip = _to_idx_point(d_mm)
        ax.annotate("", xy=tip, xytext=(fhc_idx[hu], fhc_idx[hv]), annotation_clip=False,
                    arrowprops=dict(arrowstyle="-|>", color=C_AXIS, lw=2.6, mutation_scale=18),
                    zorder=6)
    else:  # the axis is (nearly) perpendicular to this slice: mark it as out of the page
        ax.plot(fhc_idx[hu], fhc_idx[hv], marker="o", ms=17, mfc="none", mec=C_AXIS, mew=2, zorder=6)
        ax.plot(fhc_idx[hu], fhc_idx[hv], marker=".", ms=6, color=C_AXIS, zorder=6)
        ax.text(fhc_idx[hu], fhc_idx[hv] - 0.35 * r_idx, "cone axis out of page",
                color=C_AXIS, fontsize=8, ha="center", va="bottom", fontweight="bold")

    ax.set_xlim(fhc_idx[hu] - half_width, fhc_idx[hu] + half_width)
    ax.set_ylim(fhc_idx[hv] + half_width, fhc_idx[hv] - half_width)  # row index grows downward
    ax.set_xticks([]); ax.set_yticks([])
    for x, y, s, ha, va in ((0.02, 0.5, left, "left", "center"), (0.98, 0.5, right, "right", "center"),
                            (0.5, 0.98, top, "center", "top"), (0.5, 0.02, bottom, "center", "bottom")):
        ax.text(x, y, s, transform=ax.transAxes, color="white", fontsize=11,
                fontweight="bold", ha=ha, va=va)
    names = {0: "x", 1: "y", 2: "z"}
    ax.set_title(f"{title}  ({names[fixed]} = {pos}, through the FHC)\n"
                 f"{int(in_slab.sum())} rays within ±{slab:g} voxels{edge_text}", fontsize=10)
    return int(in_slab.sum())


def render(index: int, side: str, slab: float, out_png: str, radius_factor: float = 1.5,
           cone_angle: float = 45.0) -> dict:
    """Run the measurement on one hip and write the three-view figure.

    :param index: Index into the sorted NaKo segmentation files.
    :param side: 'left' or 'right' (image side).
    :param slab: Half-thickness in voxels for counting a ray as lying in a slice.
    :param out_png: Output PNG path.
    :param radius_factor: Peri-acetabular crop radius in femoral-head radii.
    :param cone_angle: Cone half-angle in degrees (the measurement's default).
    :return: A dict of the measurement statistics.
    """
    seg, mri, case_id = _load_pair(index, side)
    mean, std, mn, mx, dists, exits, hits = H.calculate_subchondral_distance_ray_tracing(
        seg, side=side, isotropic=True, cone_angle=cone_angle,
        acetabulum_radius_factor=radius_factor)

    r_idx, fhc_idx = get_femoral_head_center(seg.array, side=side, segmentation_label=1,
                                             isotropic=True)
    fhc_idx = np.asarray(fhc_idx, dtype=float)
    axis_phys = _cone_axis(seg, side, DEFAULT_AXIS, radius_factor)
    crop = _peri_acetabular_mask(seg, side, 3, radius_factor)
    exits_idx, hits_idx = _to_index(seg, exits), _to_index(seg, hits)

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.6), facecolor="white")
    for ax, view in zip(axes, VIEWS):
        _draw_panel(ax, view, seg, mri, crop, fhc_idx, float(r_idx), axis_phys, cone_angle,
                    exits_idx, hits_idx, slab, half_width=2.3 * float(r_idx))

    handles = [Line2D([], [], color=C_RAY, lw=1.8, label="ray within slab"),
               Line2D([], [], color=C_RAY, lw=0.8, alpha=0.35, label="all rays, projected"),
               Line2D([], [], ls="", marker="o", color=C_EXIT, label="head exit"),
               Line2D([], [], ls="", marker="o", color=C_HIT, label="acetabulum hit"),
               Line2D([], [], color=C_FEMUR, lw=1.6, label="femur"),
               Line2D([], [], color=C_CROP, lw=1.6, label=f"acetabulum (≤{radius_factor:g} r)"),
               Line2D([], [], color=C_PELVIS, lw=1.0, label="pelvis (full label)"),
               Line2D([], [], color="0.35", lw=1.2, ls="--", marker="+", ms=10,
                      label="femoral head (centre, radius)"),
               Line2D([], [], color=C_AXIS, lw=2.6, label=f"cone axis ({DEFAULT_AXIS})"),
               Line2D([], [], color=C_AXIS, lw=1.4, ls="--", label="cone edges (in-slice rays)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle(f"Subchondral distance — NaKo case {case_id} ({side} image side), "
                 f"cone axis '{DEFAULT_AXIS}'\n"
                 f"mean {mean:.2f} mm   std {std:.2f}   min {mn:.2f}   max {mx:.2f}   "
                 f"({len(dists)} of 200 rays reached the acetabulum)", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"case {case_id} {side}: mean={mean:.2f} std={std:.2f} rays={len(dists)}  wrote {out_png}")
    return {"case": case_id, "side": side, "mean": mean, "std": std, "rays": len(dists)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", type=int, default=0, help="index into the sorted NaKo files")
    ap.add_argument("--side", choices=["left", "right"], default="left")
    ap.add_argument("--slab", type=float, default=1.5,
                    help="half-thickness (voxels) for drawing a ray as lying in a slice")
    ap.add_argument("--out", default=None,
                    help="PNG path (default: subchondral_distance_2d_<side>.png here)")
    args = ap.parse_args()
    out = args.out or str(HERE / f"subchondral_distance_2d_{args.side}.png")
    render(args.case, args.side, args.slab, out)


if __name__ == "__main__":
    main()
