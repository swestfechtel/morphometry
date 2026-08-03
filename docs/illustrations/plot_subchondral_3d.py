#!/usr/bin/env python3
"""Render a real 3D view of the subchondral-distance ray tracing on sample data.

Loads a NaKo hip segmentation (isotropic; femur=1, cartilage=2, acetabulum=3),
splits it into image-side halves, and runs the *actual* measurement
``morphometry.measurements.hip.calculate_subchondral_distance_ray_tracing`` with
a PyVista ``Plotter`` passed as ``plot=``. The measurement itself draws the
surviving rays (red), femoral-head exit points (blue) and acetabulum hit points
(green); this script additionally adds the femur, the full pelvis label (faint)
and the peri-acetabular crop that actually aims the cone (``--radius-factor`` head
radii around the FHC — the reduced search space that isolates the acetabular
lunate surface) as translucent context, then saves an off-screen PNG (companion
to the schematic ``subchondral_distance.svg``).

    MORPH_NAKO_SAMPLE_DIR=/path/to/segmentations \
        python docs/illustrations/plot_subchondral_3d.py [--side left] [--radius-factor 1.5] [--html]

Requires the sample data (set ``MORPH_NAKO_SAMPLE_DIR``; defaults to the repo's
test path) and a PyVista off-screen backend. This is a didactic render, not a
clinical figure.
"""
import argparse
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pyvista as pv

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))  # repo root

from morphometry.hip import get_femoral_head_center  # noqa: E402
from morphometry.image_io import Segmentation  # noqa: E402
from morphometry.measurements import hip as H   # noqa: E402
from morphometry.measurements.hip import _physical_distances_from_point  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_NAKO = Path(os.environ.get(
    "MORPH_NAKO_SAMPLE_DIR", "/home/simon/Data/NaKo_sample/segmentations"))


def _sample_files() -> list[Path]:
    """The sorted NaKo segmentation files, or a fatal error if none are present."""
    files = sorted(DEFAULT_NAKO.glob("*.nii.gz"))
    if not files:
        raise SystemExit(f"No NaKo segmentations under {DEFAULT_NAKO} "
                         f"(set MORPH_NAKO_SAMPLE_DIR).")
    return files


def _load_sample(index: int = 0) -> tuple[Segmentation, str]:
    """Load the ``index``-th NaKo segmentation (LPI, de-outliered) and its case id."""
    files = _sample_files()
    path = files[index % len(files)]
    seg = Segmentation("nibabel")
    seg.read_image(str(path))
    seg.transform_coordinate_system()
    seg.remove_outliers()
    case_id = path.name.split("_")[0]  # leading numeric id, e.g. "100186"
    return seg, case_id


def _split_side(seg: Segmentation, side: str) -> Segmentation:
    """Return the requested image-side half (split at shape[0]//2), like the pipelines."""
    half = seg.array.shape[0] // 2
    arr = seg.array[:half] if side == "left" else seg.array[half:]
    return Segmentation.from_nibabel(nib.Nifti1Image(arr, seg.affine, seg.header))


def _surface_from_mask(image: Segmentation, mask: np.ndarray) -> pv.PolyData | None:
    """Marching-cubes surface of a boolean mask, in the same physical space as the rays.

    Returns None if the mask is empty. Vertices are mapped index->physical via the
    image transform so the mesh registers with the ray endpoints (which the
    measurement also emits in physical/mm coordinates).
    """
    mask = mask.astype(np.uint8)
    if mask.sum() == 0:
        return None
    grid = pv.ImageData(dimensions=np.array(mask.shape) + 1)
    grid.cell_data["v"] = mask.flatten(order="F")
    surf = grid.cast_to_unstructured_grid().extract_cells(
        np.flatnonzero(mask.flatten(order="F"))).extract_surface()
    pts = np.array([image.transform_index_to_physical_point(p) for p in surf.points])
    surf.points = pts
    return surf.smooth(n_iter=40)


def _peri_acetabular_mask(image: Segmentation, side: str, label: int,
                          radius_factor: float) -> np.ndarray:
    """The acetabulum label restricted to voxels within radius_factor head radii of the FHC.

    Reproduces the peri-acetabular crop the measurement now applies, so the render
    can show the reduced search space that actually aims the cone.
    """
    r_idx, fhc_idx = get_femoral_head_center(image.array, side=side, segmentation_label=1,
                                             isotropic=True)
    fhc_phys = np.asarray(image.transform_index_to_physical_point(fhc_idx), dtype=float)
    surf_phys = np.asarray(
        image.transform_index_to_physical_point(np.asarray(fhc_idx) + np.array([r_idx, 0, 0])),
        dtype=float)
    r_mm = float(np.linalg.norm(surf_phys - fhc_phys))

    mask = image.array == label
    voxels = np.argwhere(mask)
    dist = _physical_distances_from_point(image, voxels, fhc_idx)
    keep = voxels[dist <= radius_factor * r_mm]
    restricted = np.zeros_like(mask)
    restricted[keep[:, 0], keep[:, 1], keep[:, 2]] = True
    return restricted


def render_case(side_img: Segmentation, case_id: str, side: str, radius_factor: float | None,
                out_png: str | None, *, n_rays: int = 200, cone_angle: float = 45.0,
                html: bool = False, interactive: bool = False) -> dict:
    """Render one hip's ray tracing and return its distance statistics.

    Runs the real ``calculate_subchondral_distance_ray_tracing`` with a PyVista
    ``Plotter`` (which draws the rays/exit/hit markers), layers on the femur, full
    pelvis and peri-acetabular-crop surfaces, frames the joint, and writes ``out_png``
    (plus an interactive ``.html`` when ``html``). Returns the mean/std/min/max/ray
    count so batch callers can tabulate across cases.
    :param side_img: A single image-side hip Segmentation (femur=1, acetabulum=3).
    :param case_id: A short id used in the on-canvas title.
    :param side: 'left' or 'right' (image side).
    :param radius_factor: Peri-acetabular crop radius in head radii, or None for the full label.
    :param out_png: Where to write the PNG (ignored when ``interactive``).
    :param n_rays: Rays cast within the cone.
    :param cone_angle: Cone half-angle in degrees.
    :param html: Also export a rotatable ``.html`` next to the PNG.
    :param interactive: Open a live window instead of writing files.
    :return: A dict with keys ``case, side, mean, std, min, max, rays``.
    """
    pv.set_plot_theme("document")
    plotter = pv.Plotter(off_screen=not interactive, window_size=(1100, 900))

    # context surfaces (added first so the rays draw on top): femur + the full pelvis
    # (faint) with the peri-acetabular crop that actually aims the cone drawn on top
    femur = _surface_from_mask(side_img, side_img.array == 1)
    pelvis = _surface_from_mask(side_img, side_img.array == 3)
    if femur is not None:
        plotter.add_mesh(femur, color="#ECE3D0", opacity=0.25, smooth_shading=True, label="femur")
    if pelvis is not None:
        plotter.add_mesh(pelvis, color="#C9B37E", opacity=0.18, smooth_shading=True,
                         label="pelvis (full label)")
    if radius_factor is not None:
        peri = _surface_from_mask(side_img, _peri_acetabular_mask(side_img, side, 3, radius_factor))
        if peri is not None:
            plotter.add_mesh(peri, color="#B8860B", opacity=0.55, smooth_shading=True,
                             label=f"acetabulum (≤{radius_factor:g} r)")

    # the real measurement draws the rays / exit (blue) / hit (green) onto `plot`
    mean, std, mn, mx, dists, exits, hits = H.calculate_subchondral_distance_ray_tracing(
        side_img, side=side, isotropic=True, n_rays=n_rays,
        cone_angle=cone_angle, acetabulum_radius_factor=radius_factor, plot=plotter)

    factor_txt = f"{radius_factor:g} r" if radius_factor is not None else "full"
    plotter.add_text(
        f"case {case_id} ({side}) — crop {factor_txt} — {len(dists)} rays\n"
        f"mean {mean:.2f} mm  (std {std:.2f}, min {mn:.2f}, max {mx:.2f})",
        font_size=12, position="upper_left")
    legend = [["femur", "#ECE3D0"], ["pelvis (full label)", "#C9B37E"]]
    if radius_factor is not None:
        legend.append([f"acetabulum (≤{radius_factor:g} r)", "#B8860B"])
    legend += [["ray", "#DC2626"], ["head exit", "#2563EB"], ["acetabulum hit", "#0E9F6E"]]
    plotter.add_legend(labels=legend, bcolor="white", size=(0.24, 0.18),
                       loc="lower left", face="circle")

    # frame the joint: centre on the ray cloud, not the whole hemipelvis
    joint_centre = np.vstack([exits, hits]).mean(axis=0)
    plotter.camera_position = "yz"
    plotter.camera.azimuth = 35
    plotter.camera.elevation = 20
    plotter.reset_camera()
    plotter.camera.focal_point = tuple(joint_centre)
    plotter.camera.zoom(2.2)

    print(f"case {case_id} {side}: mean={mean:.2f} mm  std={std:.2f}  min={mn:.2f}  "
          f"max={mx:.2f}  rays={len(dists)}")
    if interactive:
        plotter.show()
    else:
        if html and out_png:
            html_path = str(Path(out_png).with_suffix(".html"))
            plotter.export_html(html_path)
            print("wrote", html_path)
        plotter.show(screenshot=out_png)
        print("wrote", out_png)
    return {"case": case_id, "side": side, "mean": mean, "std": std,
            "min": mn, "max": mx, "rays": len(dists)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", choices=["left", "right"], default="left")
    ap.add_argument("--case", type=int, default=0,
                    help="index into the sorted NaKo sample files (default 0)")
    ap.add_argument("--out", default=None,
                    help="PNG path (default: subchondral_distance_3d_<side>.png)")
    ap.add_argument("--html", action="store_true",
                    help="also export an interactive, rotatable HTML next to the PNG")
    ap.add_argument("--n-rays", type=int, default=200)
    ap.add_argument("--cone-angle", type=float, default=45.0)
    ap.add_argument("--radius-factor", type=float, default=1.5,
                    help="peri-acetabular crop radius in femoral-head radii (None-like <=0 disables)")
    ap.add_argument("--interactive", action="store_true",
                    help="open a window instead of rendering off-screen")
    args = ap.parse_args()
    out_png = args.out or str(HERE / f"subchondral_distance_3d_{args.side}.png")
    radius_factor = args.radius_factor if args.radius_factor > 0 else None

    seg, case_id = _load_sample(args.case)
    side_img = _split_side(seg, args.side)
    render_case(side_img, case_id, args.side, radius_factor, out_png,
                n_rays=args.n_rays, cone_angle=args.cone_angle,
                html=args.html, interactive=args.interactive)


if __name__ == "__main__":
    main()
