#!/usr/bin/env python3
"""Render a real 3D view of the subchondral-distance ray tracing on sample data.

Loads a NaKo hip segmentation (isotropic; femur=1, cartilage=2, acetabulum=3),
splits it into image-side halves, and runs the *actual* measurement
``morphometry.measurements.hip.calculate_subchondral_distance_ray_tracing`` with
a PyVista ``Plotter`` passed as ``plot=``. The measurement itself draws the
surviving rays (red), femoral-head exit points (blue) and acetabulum hit points
(green); this script additionally adds the femur, the full pelvis label (faint),
the peri-acetabular crop that restricts the hit search (``--radius-factor`` head
radii around the FHC) and an arrow showing the cone axis itself, then saves an
off-screen PNG (companion to the schematic ``subchondral_distance.svg``).

``--axis`` selects how the ray cone is aimed, mirroring the measurement's
``cone_axis`` argument. Its default is read off the library signature, so this
script always illustrates whatever the measurement actually does by default:

* ``superior`` (the library default) — straight up along world ``+z``, a fixed
  and therefore standardised axis that samples the weight-bearing acetabular
  dome in every subject.
* ``acetabulum`` (legacy) — FHC -> peri-acetabular centroid, following the
  subject's own acetabulum. On this sample data it sits ~46 degrees medial of
  superior and sprays roughly half its rays into the non-articular acetabular
  fossa, which is why it is no longer the default.
* ``compare`` — both of the above side by side in one linked-camera scene.

The output PNG/HTML for the library-default axis is unsuffixed
(``subchondral_distance_3d_<side>.png``); any non-default axis gets an explicit
``_<axis>`` suffix, so the plain filename always shows the default behaviour.

``--table N`` skips rendering and instead sweeps both axis modes over the first
``N`` sample cases x both image sides, printing and writing a CSV comparison.

    MORPH_NAKO_SAMPLE_DIR=/path/to/segmentations \
        python docs/illustrations/plot_subchondral_3d.py \
            [--side left] [--axis compare] [--radius-factor 1.5] [--html]
    MORPH_NAKO_SAMPLE_DIR=... python docs/illustrations/plot_subchondral_3d.py --table 8

Requires the sample data (set ``MORPH_NAKO_SAMPLE_DIR``; defaults to the repo's
test path) and a PyVista off-screen backend. This is a didactic render, not a
clinical figure.
"""
import argparse
import csv
import inspect
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
from morphometry.measurements.hip import _physical_distances_from_point, _superior_direction  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_NAKO = Path(os.environ.get(
    "MORPH_NAKO_SAMPLE_DIR", "/home/simon/Data/NaKo_sample/segmentations"))
DEFAULT_TABLE_CSV = ("/tmp/claude-1000/-home-simon-Work-morphometry/"
                     "30dd999e-8834-4902-a6f0-51b2cc0935f1/scratchpad/"
                     "subchondral_axis_comparison.csv")

# one colour per cone-axis mode, so the two arrows stay identifiable when both are
# drawn in the same panel (and distinct from ray red / exit blue / hit green)
AXIS_COLORS = {"acetabulum": "#7C3AED", "superior": "#0891B2"}

# The library's own ``cone_axis`` default, read off the signature rather than hard-coded:
# this script exists to illustrate the measurement, so its --axis default (and hence which
# render gets the unsuffixed file name) must not be able to drift away from it.
DEFAULT_AXIS = inspect.signature(
    H.calculate_subchondral_distance_ray_tracing).parameters["cone_axis"].default


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


def _femoral_head(image: Segmentation, side: str, femur_label: int = 1) -> tuple[np.ndarray, float]:
    """The femoral head center in physical space and its radius in mm.

    Uses the same ``get_femoral_head_center`` call (isotropic sample data) as the
    measurement, then converts the index-space radius to mm by stepping one radius
    along index axis 0 and measuring the physical offset.
    :param image: A single image-side hip Segmentation.
    :param side: 'left' or 'right' (image side).
    :param femur_label: The label of the femur in the segmentation mask.
    :return: A tuple ``(fhc_physical, radius_mm)``.
    """
    r_idx, fhc_idx = get_femoral_head_center(image.array, side=side,
                                             segmentation_label=femur_label, isotropic=True)
    fhc_phys = np.asarray(image.transform_index_to_physical_point(fhc_idx), dtype=float)
    surf_phys = np.asarray(
        image.transform_index_to_physical_point(np.asarray(fhc_idx) + np.array([r_idx, 0, 0])),
        dtype=float)
    return fhc_phys, float(np.linalg.norm(surf_phys - fhc_phys))


def _peri_acetabular_mask(image: Segmentation, side: str, label: int,
                          radius_factor: float) -> np.ndarray:
    """The acetabulum label restricted to voxels within radius_factor head radii of the FHC.

    Reproduces the peri-acetabular crop the measurement applies, so the render can
    show the reduced search space that both aims the cone (in ``acetabulum`` mode)
    and restricts the acetabulum hit search (in both modes).
    """
    _, fhc_idx = get_femoral_head_center(image.array, side=side, segmentation_label=1,
                                         isotropic=True)
    _, r_mm = _femoral_head(image, side)

    mask = image.array == label
    voxels = np.argwhere(mask)
    dist = _physical_distances_from_point(image, voxels, fhc_idx)
    keep = voxels[dist <= radius_factor * r_mm]
    restricted = np.zeros_like(mask)
    restricted[keep[:, 0], keep[:, 1], keep[:, 2]] = True
    return restricted


def _cone_axis(image: Segmentation, side: str, axis_mode: str,
               radius_factor: float | None, acetabulum_label: int = 3) -> np.ndarray:
    """The unit cone axis the measurement would use, in physical space.

    Mirrors ``_subchondral_distance_ray_tracing``: ``'superior'`` delegates to
    :func:`morphometry.measurements.hip._superior_direction`, ``'acetabulum'``
    points from the FHC at the centroid of the (optionally cropped) acetabulum
    label. Shared with the measurement rather than reimplemented, so the drawn
    arrow cannot drift from the rays.
    :param image: A single image-side hip Segmentation.
    :param side: 'left' or 'right' (image side).
    :param axis_mode: 'superior' or 'acetabulum'.
    :param radius_factor: Peri-acetabular crop radius in head radii, or None.
    :param acetabulum_label: The label of the acetabulum / pelvis.
    :return: A (3,) unit vector in physical (mm) space.
    """
    if axis_mode == "superior":
        return _superior_direction()

    if radius_factor is not None:
        mask = _peri_acetabular_mask(image, side, acetabulum_label, radius_factor)
    else:
        mask = image.array == acetabulum_label
    centroid_idx = np.argwhere(mask).mean(axis=0)
    centroid_phys = np.asarray(image.transform_index_to_physical_point(centroid_idx), dtype=float)
    fhc_phys, _ = _femoral_head(image, side)
    axis = centroid_phys - fhc_phys
    return axis / np.linalg.norm(axis)


def _anterior_direction(image: Segmentation) -> np.ndarray:
    """The unit physical-space vector pointing anteriorly, from the image's own affine.

    ``_load_sample`` reorients every sample to LPI, so index axis 1 increases toward
    *Posterior* and anterior is the physical direction of decreasing index axis 1.
    Unlike ``morphometry.measurements.hip._superior_direction`` — which is a world
    ``+z`` constant precisely so it does not depend on that reorientation — this one
    reads the affine, which is safe here because it only disambiguates the sign of the
    camera direction (so both hips are viewed from the same side of the body). Getting
    it wrong would mirror the view, not change a measurement.
    :param image: A Segmentation providing the index<->physical transform.
    :return: A (3,) unit vector in physical (mm) space pointing anteriorly.
    """
    p0 = np.asarray(image.transform_index_to_physical_point([0, 0, 0]), dtype=float)
    p1 = np.asarray(image.transform_index_to_physical_point([0, 1, 0]), dtype=float)
    direction = -(p1 - p0)
    return direction / np.linalg.norm(direction)


def _axis_angle(a: np.ndarray, b: np.ndarray) -> float:
    """The angle in degrees between two (unit) direction vectors."""
    return float(np.degrees(np.arccos(np.clip(float(np.dot(a, b)), -1.0, 1.0))))


def _add_axis_arrow(plotter: pv.Plotter, start: np.ndarray, direction: np.ndarray,
                    r_mm: float, color: str, *, active: bool) -> None:
    """Draw one cone axis as an arrow from the FHC.

    The panel's own axis is drawn solid and thick; the other mode's axis is drawn
    thin and translucent, as a reference the reader can compare the tilt against.
    :param plotter: The PyVista Plotter (its currently active subplot is used).
    :param start: The arrow origin (the FHC) in physical space.
    :param direction: The unit axis direction in physical space.
    :param r_mm: The femoral head radius in mm, used to scale the arrow.
    :param color: The arrow colour (see ``AXIS_COLORS``).
    :param active: Whether this is the panel's own axis (solid) or the reference (faint).
    """
    scale = 2.2 * r_mm if active else 1.9 * r_mm
    arrow = pv.Arrow(start=start, direction=direction, scale=scale, tip_length=0.16,
                     tip_radius=0.045 if active else 0.028,
                     shaft_radius=0.016 if active else 0.008)
    plotter.add_mesh(arrow, color=color, opacity=1.0 if active else 0.45,
                     ambient=0.35 if active else 0.2)


def _add_scene(plotter: pv.Plotter, side_img: Segmentation, case_id: str, side: str,
               axis_mode: str, radius_factor: float | None, *, n_rays: int,
               cone_angle: float, legend: bool = True,
               reference_mode: str | None = None) -> dict:
    """Populate the active (sub)plot with one axis variant of the ray tracing.

    Adds the femur / full pelvis / peri-acetabular context surfaces, runs the real
    measurement with this plotter as ``plot=`` (which draws the rays and the exit /
    hit markers), draws the cone axis (and optionally the other mode's axis, faint)
    as arrows from the FHC, and writes a text block with the distance statistics.

    The peri-acetabular crop surface is clipped on the camera's near side and kept
    translucent: at 0.55 opacity it sat in front of the ray fan and hid the very
    thing the figure is about. Clipping needs the view direction, so the caller
    passes it in via ``_scene_view_direction``.
    :param plotter: The PyVista Plotter (its currently active subplot is used).
    :param side_img: A single image-side hip Segmentation (femur=1, acetabulum=3).
    :param case_id: A short id used in the on-canvas title.
    :param side: 'left' or 'right' (image side).
    :param axis_mode: 'superior' or 'acetabulum' — the measurement's ``cone_axis``.
    :param radius_factor: Peri-acetabular crop radius in head radii, or None for the full label.
    :param n_rays: Rays cast within the cone.
    :param cone_angle: Cone half-angle in degrees.
    :param legend: Whether to draw the colour legend in this subplot.
    :param reference_mode: The other axis mode to draw as a faint reference arrow (and
        to report the inter-axis angle against), or None to draw only this panel's axis.
    :return: A dict with keys ``case, side, axis, mean, std, min, max, rays, angle,
        axis_vec, fhc, radius_mm, exits, hits``.
    """
    fhc_phys, r_mm = _femoral_head(side_img, side)
    axis_vec = _cone_axis(side_img, side, axis_mode, radius_factor)
    view_dir = _scene_view_direction(side_img, side, radius_factor)

    # context surfaces (added first so the rays draw on top): femur + the full pelvis
    # (faint) with the peri-acetabular crop that restricts the hit search drawn on top.
    # The crop is clipped to the far side of the FHC so it backs the rays instead of
    # covering them; the femur stays whole (it is faint enough to see through).
    femur = _surface_from_mask(side_img, side_img.array == 1)
    pelvis = _surface_from_mask(side_img, side_img.array == 3)
    if femur is not None:
        plotter.add_mesh(femur, color="#ECE3D0", opacity=0.18, smooth_shading=True, label="femur")
    if pelvis is not None:
        plotter.add_mesh(pelvis, color="#C9B37E", opacity=0.12, smooth_shading=True,
                         label="pelvis (full label)")
    if radius_factor is not None:
        peri = _surface_from_mask(side_img, _peri_acetabular_mask(side_img, side, 3, radius_factor))
        if peri is not None:
            far_half = peri.clip(normal=-view_dir, origin=fhc_phys, invert=False)
            plotter.add_mesh(far_half if far_half.n_points else peri, color="#B8860B",
                             opacity=0.32, smooth_shading=True,
                             label=f"acetabulum (≤{radius_factor:g} r)")

    # the real measurement draws the rays / exit (blue) / hit (green) onto `plot`
    mean, std, mn, mx, dists, exits, hits = H.calculate_subchondral_distance_ray_tracing(
        side_img, side=side, isotropic=True, n_rays=n_rays, cone_angle=cone_angle,
        acetabulum_radius_factor=radius_factor, cone_axis=axis_mode, plot=plotter)

    # the cone axes as arrows from the FHC (same derivation as the measurement)
    angle = None
    if reference_mode is not None:
        reference_vec = _cone_axis(side_img, side, reference_mode, radius_factor)
        angle = _axis_angle(axis_vec, reference_vec)
        _add_axis_arrow(plotter, fhc_phys, reference_vec, r_mm,
                        AXIS_COLORS[reference_mode], active=False)
    _add_axis_arrow(plotter, fhc_phys, axis_vec, r_mm, AXIS_COLORS[axis_mode], active=True)

    factor_txt = f"{radius_factor:g} r" if radius_factor is not None else "full"
    text = (f"case {case_id} ({side}) — {axis_mode} axis — crop {factor_txt} — {len(dists)} rays\n"
            f"mean {mean:.2f} mm  (std {std:.2f}, min {mn:.2f}, max {mx:.2f})")
    if angle is not None:
        text += f"\naxes {angle:.1f}° apart ({reference_mode} axis shown faint)"
    plotter.add_text(text, font_size=12, position="upper_left")
    if legend:
        labels = [["femur", "#ECE3D0"], ["pelvis (full label)", "#C9B37E"]]
        if radius_factor is not None:
            labels.append([f"acetabulum (≤{radius_factor:g} r)", "#B8860B"])
        labels += [["ray", "#DC2626"], ["head exit", "#2563EB"], ["acetabulum hit", "#0E9F6E"]]
        if reference_mode is not None:
            labels += [[f"{m} axis", AXIS_COLORS[m]] for m in ("acetabulum", "superior")]
        else:
            labels.append([f"{axis_mode} axis (cone)", AXIS_COLORS[axis_mode]])
        plotter.add_legend(labels=labels, bcolor="white", size=(0.23, 0.18),
                           loc="lower left", face="circle")

    print(f"case {case_id} {side} [{axis_mode}]: mean={mean:.2f} mm  std={std:.2f}  "
          f"min={mn:.2f}  max={mx:.2f}  rays={len(dists)}"
          + (f"  angle_to_{reference_mode}={angle:.1f} deg" if angle is not None else ""))
    return {"case": case_id, "side": side, "axis": axis_mode, "mean": mean, "std": std,
            "min": mn, "max": mx, "rays": len(dists), "angle": angle, "axis_vec": axis_vec,
            "fhc": fhc_phys, "radius_mm": r_mm, "exits": exits, "hits": hits}


def _scene_view_direction(side_img: Segmentation, side: str,
                          radius_factor: float | None) -> np.ndarray:
    """A viewing direction roughly perpendicular to *both* cone axes.

    The figure exists to show the angle between the acetabulum and superior axes, so
    the camera must not look down either of them (the default oblique view did, and
    the ray fan collapsed into a flat disc of dots). The normal of the plane the two
    axes span puts both of them in the image plane, showing the full tilt and the
    cone opening. The sign is chosen so the camera sits anteriorly, which keeps left
    and right hips consistent; near-parallel axes fall back to a plain anterior view.
    :param side_img: A single image-side hip Segmentation.
    :param side: 'left' or 'right' (image side).
    :param radius_factor: Peri-acetabular crop radius in head radii, or None.
    :return: A (3,) unit vector pointing from the joint toward the camera.
    """
    anterior = _anterior_direction(side_img)
    normal = np.cross(_cone_axis(side_img, side, "acetabulum", radius_factor),
                      _superior_direction())
    if np.linalg.norm(normal) < 1e-6:  # axes (anti)parallel: no plane to be normal to
        return anterior
    normal = normal / np.linalg.norm(normal)
    return normal if float(np.dot(normal, anterior)) > 0 else -normal


def _frame_joint(plotter: pv.Plotter, points: np.ndarray, view_dir: np.ndarray,
                 up: np.ndarray) -> None:
    """Look at the ray cloud from ``view_dir``, superior up, and zoom in on the joint.

    ``reset_camera`` keeps the direction and up vector set here and only fixes the
    distance to the scene bounds, so the framing is driven by the anatomy rather
    than by hard-coded azimuth/elevation.
    :param plotter: The PyVista Plotter (its currently active subplot is used).
    :param points: The point cloud to centre on (the ray exit/hit points).
    :param view_dir: Unit vector from the joint toward the camera.
    :param up: Unit vector to put at the top of the frame (the superior direction).
    """
    focal = points.mean(axis=0)
    plotter.camera_position = [tuple(focal + view_dir), tuple(focal), tuple(up)]
    plotter.reset_camera()
    plotter.camera.focal_point = tuple(focal)
    plotter.camera.zoom(3.4)  # the scene bounds are the whole hemipelvis; the joint is small


def _clean(stats: dict) -> dict:
    """Drop the bulky/vector entries from a scene's statistics, for terminal reporting."""
    return {k: v for k, v in stats.items()
            if k not in ("exits", "hits", "axis_vec", "fhc", "radius_mm")}


def _finish(plotter: pv.Plotter, out_png: str | None, html: bool, interactive: bool) -> None:
    """Show the plotter interactively, or export the HTML and write the screenshot."""
    if interactive:
        plotter.show()
        return
    if html and out_png:
        html_path = str(Path(out_png).with_suffix(".html"))
        plotter.export_html(html_path)
        print("wrote", html_path)
    plotter.show(screenshot=out_png)
    print("wrote", out_png)


def render_case(side_img: Segmentation, case_id: str, side: str, radius_factor: float | None,
                out_png: str | None, *, axis_mode: str = DEFAULT_AXIS, n_rays: int = 200,
                cone_angle: float = 45.0, html: bool = False,
                interactive: bool = False) -> dict:
    """Render one hip's ray tracing for a single cone-axis mode and return its statistics.

    Runs the real ``calculate_subchondral_distance_ray_tracing`` with a PyVista
    ``Plotter`` (which draws the rays/exit/hit markers), layers on the femur, full
    pelvis and peri-acetabular-crop surfaces plus the cone-axis arrow, frames the
    joint, and writes ``out_png`` (plus an interactive ``.html`` when ``html``).
    Returns the mean/std/min/max/ray count so batch callers can tabulate across cases.
    :param side_img: A single image-side hip Segmentation (femur=1, acetabulum=3).
    :param case_id: A short id used in the on-canvas title.
    :param side: 'left' or 'right' (image side).
    :param radius_factor: Peri-acetabular crop radius in head radii, or None for the full label.
    :param out_png: Where to write the PNG (ignored when ``interactive``).
    :param axis_mode: 'superior' or 'acetabulum' — the measurement's ``cone_axis``.
        Defaults to ``DEFAULT_AXIS``, i.e. the library's own ``cone_axis`` default, so a
        caller that does not pass one (``render_case_gallery.py``) renders default behaviour.
    :param n_rays: Rays cast within the cone.
    :param cone_angle: Cone half-angle in degrees.
    :param html: Also export a rotatable ``.html`` next to the PNG.
    :param interactive: Open a live window instead of writing files.
    :return: A dict with keys ``case, side, axis, mean, std, min, max, rays``.
    """
    pv.set_plot_theme("document")
    plotter = pv.Plotter(off_screen=not interactive, window_size=(1100, 900))
    stats = _add_scene(plotter, side_img, case_id, side, axis_mode, radius_factor,
                       n_rays=n_rays, cone_angle=cone_angle)
    _frame_joint(plotter, np.vstack([stats["exits"], stats["hits"]]),
                 _scene_view_direction(side_img, side, radius_factor),
                 _superior_direction())
    _finish(plotter, out_png, html, interactive)
    return _clean(stats)


def render_comparison(side_img: Segmentation, case_id: str, side: str,
                      radius_factor: float | None, out_png: str | None, *, n_rays: int = 200,
                      cone_angle: float = 45.0, html: bool = False,
                      interactive: bool = False) -> list[dict]:
    """Render the acetabulum- and superior-axis variants side by side in one scene.

    Both subplots get the same femur / full-pelvis / peri-acetabular context, the
    measurement's own ray overlay, the cone-axis arrow and a statistics text block;
    the two cameras are linked so the reader compares the same viewpoint.
    :param side_img: A single image-side hip Segmentation (femur=1, acetabulum=3).
    :param case_id: A short id used in the on-canvas titles.
    :param side: 'left' or 'right' (image side).
    :param radius_factor: Peri-acetabular crop radius in head radii, or None for the full label.
    :param out_png: Where to write the PNG (ignored when ``interactive``).
    :param n_rays: Rays cast within the cone.
    :param cone_angle: Cone half-angle in degrees.
    :param html: Also export a rotatable ``.html`` next to the PNG.
    :param interactive: Open a live window instead of writing files.
    :return: A list of two statistics dicts, acetabulum axis first.
    """
    pv.set_plot_theme("document")
    plotter = pv.Plotter(off_screen=not interactive, shape=(1, 2), window_size=(1900, 900))

    stats = []
    modes = ("acetabulum", "superior")
    for column, axis_mode in enumerate(modes):
        plotter.subplot(0, column)
        stats.append(_add_scene(plotter, side_img, case_id, side, axis_mode, radius_factor,
                                n_rays=n_rays, cone_angle=cone_angle, legend=True,
                                reference_mode=modes[1 - column]))

    plotter.link_views()
    plotter.subplot(0, 0)
    _frame_joint(plotter, np.vstack([s["exits"] for s in stats] + [s["hits"] for s in stats]),
                 _scene_view_direction(side_img, side, radius_factor),
                 _superior_direction())
    _finish(plotter, out_png, html, interactive)
    return [_clean(s) for s in stats]


def run_table(n_cases: int, radius_factor: float | None, csv_path: str, *,
              n_rays: int = 200, cone_angle: float = 45.0) -> list[dict]:
    """Sweep both cone-axis modes over the first ``n_cases`` samples x both image sides.

    No rendering happens; only the measurement is run. Per-case failures are caught
    and recorded as NaN (repo convention for batch scripts) so one bad segmentation
    does not abort the sweep. The table is printed and written as CSV.
    :param n_cases: How many of the sorted sample files to process.
    :param radius_factor: Peri-acetabular crop radius in head radii, or None for the full label.
    :param csv_path: Where to write the CSV.
    :param n_rays: Rays cast within the cone.
    :param cone_angle: Cone half-angle in degrees.
    :return: One row dict per (case, side) with both modes' statistics and the mean delta.
    """
    files = _sample_files()
    rows = []
    for index in range(min(n_cases, len(files))):
        seg, case_id = _load_sample(index)
        for side in ("left", "right"):
            row = {"case": case_id, "side": side}
            for axis_mode in ("acetabulum", "superior"):
                try:
                    side_img = _split_side(seg, side)
                    mean, std, _, _, dists, _, _ = H.calculate_subchondral_distance_ray_tracing(
                        side_img, side=side, isotropic=True, n_rays=n_rays,
                        cone_angle=cone_angle, acetabulum_radius_factor=radius_factor,
                        cone_axis=axis_mode)
                    row[f"{axis_mode}_mean"] = mean
                    row[f"{axis_mode}_std"] = std
                    row[f"{axis_mode}_rays"] = len(dists)
                except (RuntimeError, AssertionError, ValueError) as exc:
                    print(f"case {case_id} {side} [{axis_mode}] failed: {exc}")
                    row[f"{axis_mode}_mean"] = np.nan
                    row[f"{axis_mode}_std"] = np.nan
                    row[f"{axis_mode}_rays"] = np.nan
            row["delta_mean"] = row["superior_mean"] - row["acetabulum_mean"]
            rows.append(row)

    header = ["case", "side", "acetabulum_mean", "acetabulum_std", "acetabulum_rays",
              "superior_mean", "superior_std", "superior_rays", "delta_mean"]
    widths = {"case": 8, "side": 6}
    line = "  ".join(f"{c:>{widths.get(c, 16)}}" for c in header)
    print("\n" + line)
    print("-" * len(line))
    for row in rows:
        cells = []
        for column in header:
            value = row[column]
            if isinstance(value, str):
                cells.append(f"{value:>{widths.get(column, 16)}}")
            elif column.endswith("_rays"):
                cells.append(f"{value:>{widths.get(column, 16)}.0f}")
            else:
                cells.append(f"{value:>{widths.get(column, 16)}.2f}")
        print("  ".join(cells))

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print("\nwrote", csv_path)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", choices=["left", "right"], default="left")
    ap.add_argument("--axis", choices=["acetabulum", "superior", "compare"],
                    default=DEFAULT_AXIS,
                    help=f"cone axis: straight superior, the legacy acetabulum-centroid "
                         f"axis, or both side by side (default: {DEFAULT_AXIS}, i.e. the "
                         f"library's own cone_axis default)")
    ap.add_argument("--case", type=int, default=0,
                    help="index into the sorted NaKo sample files (default 0)")
    ap.add_argument("--out", default=None,
                    help=f"PNG path (default: subchondral_distance_3d_<side>.png for the "
                         f"library-default {DEFAULT_AXIS} axis, "
                         f"subchondral_distance_3d_<side>_<axis>.png otherwise, "
                         f"or subchondral_axis_comparison_<side>.png for --axis compare)")
    ap.add_argument("--html", action="store_true",
                    help="also export an interactive, rotatable HTML next to the PNG")
    ap.add_argument("--n-rays", type=int, default=200)
    ap.add_argument("--cone-angle", type=float, default=45.0)
    ap.add_argument("--radius-factor", type=float, default=1.5,
                    help="peri-acetabular crop radius in femoral-head radii (None-like <=0 disables)")
    ap.add_argument("--interactive", action="store_true",
                    help="open a window instead of rendering off-screen")
    ap.add_argument("--table", type=int, default=None, metavar="N",
                    help="run both axis modes over the first N sample cases x both sides "
                         "(no rendering) and print/write a comparison table")
    ap.add_argument("--table-csv", default=DEFAULT_TABLE_CSV,
                    help="where --table writes its CSV")
    args = ap.parse_args()
    radius_factor = args.radius_factor if args.radius_factor > 0 else None

    if args.table is not None:
        run_table(args.table, radius_factor, args.table_csv,
                  n_rays=args.n_rays, cone_angle=args.cone_angle)
        return

    seg, case_id = _load_sample(args.case)
    side_img = _split_side(seg, args.side)

    if args.axis == "compare":
        out_png = args.out or str(HERE / f"subchondral_axis_comparison_{args.side}.png")
        render_comparison(side_img, case_id, args.side, radius_factor, out_png,
                          n_rays=args.n_rays, cone_angle=args.cone_angle,
                          html=args.html, interactive=args.interactive)
        return

    # the library-default axis owns the plain file name, so the unsuffixed render always
    # depicts what the measurement does by default; other modes are suffixed explicitly
    suffix = "" if args.axis == DEFAULT_AXIS else f"_{args.axis}"
    out_png = args.out or str(HERE / f"subchondral_distance_3d_{args.side}{suffix}.png")
    render_case(side_img, case_id, args.side, radius_factor, out_png,
                axis_mode=args.axis, n_rays=args.n_rays, cone_angle=args.cone_angle,
                html=args.html, interactive=args.interactive)


if __name__ == "__main__":
    main()
