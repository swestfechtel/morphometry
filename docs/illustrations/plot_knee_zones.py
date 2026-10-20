#!/usr/bin/env python3
"""Render the Wirth & Eckstein 2008 knee cartilage subregions on real segmentations.

Companion to ``docs/wirth_eckstein_2008_plan.md`` steps 1-5: a set of off-screen
PyVista figures that show what the subdivision actually does, drawn with the
library's own plotting API (``Tibia.plot_regions`` / ``Femur.plot_regions`` /
``plot_knee_segments`` / ``plot_knee_thickness`` / ``plot_thinnest_point``) and the
library's own colour tables (``TIBIA_SUBREGIONS`` / ``FEMUR_SUBREGIONS``), so the
figures cannot drift from the code that produces the numbers. Every tunable the
figures depend on (the central share, the weight-bearing fraction, the bone labels)
is read off the library signatures rather than repeated here.

Figures (``--figures``, default ``all``), written to ``--out`` (default
``out/knee_zones``):

* ``tibia``   — the tibial tAB subdivision seen down each plate's own normal
  (superior, anterior up), with the central ellipse, the two +-45 degree wedge
  planes, the internal / anterior directions and one label per region
  (paper Fig. 5A).
* ``femur``   — the femoral tAB subdivision from inferior and from posterior
  (paper Fig. 6A), with the femoral frame's axes, the trochlear notch, the notch
  plane and each condyle's 60% posterior border, plus the thirds labelled.
* ``both``    — both bones in one scene: the labelled tAB on the left, the
  cartilage voxels the labels were transferred to on the right.
* ``thickness`` — each tAB face-on as an area-weighted surface thickness heatmap,
  both panels on one shared colour scale starting at 0 mm (paper Fig. 4B/5B), plus
  a second figure that paints the 0 mm denuded / rim triangles magenta so they are
  visible rather than lost in the dark end of the colour map, and (as HTML) the
  whole joint in one rotatable scene.
* ``flexion`` — the same knee at 0 and 90 degrees of flexion side by side: the raw
  pose (how different the two scans are), then the tibial and femoral subdivisions
  in their own bone frames with the region shares printed, which is the visual
  proof that the regions no longer move with flexion.
* ``thinnest`` — the thinnest-point arrows of the whole knee and of the two
  compartments (:func:`find_thinnest_area_by_compartment`).

**Region names are the patient-side keys the code uses** (``clt`` ... ``prt``,
``iclf`` ... ``prf``). They are *not* relabelled medial / lateral: the ``dfg_knees``
laterality is unrecorded (plan open issue 3). ``--laterality`` passes an explicit
assumption through to the library, which then adds the paper's medial/lateral names
in brackets; the figure titles mark it as an assumption.

    python docs/illustrations/plot_knee_zones.py \
        [--data-dir /home/simon/Data/dfg_knees/monai_segmentations/labels/original] \
        [--scan K1_0_unbelastet] [--flexion-scan K1_90_unbelastet] \
        [--figures tibia femur] [-o out/knee_zones] [--size 1600 1200] [--no-html]

Needs the ``dfg_knees`` segmentations (labels: 1 femur bone, 2 tibia bone,
3 femoral cartilage, 4 tibial cartilage) and a PyVista off-screen backend; no
display is required. ``K7_90_belastet`` / ``K8_90_belastet`` /
``K11_90_belastet`` are QC-flagged (see CLAUDE.md) and are refused unless
``--allow-flagged`` is given. Didactic renders, not clinical figures.
"""
import argparse
import inspect
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

# Off-screen before PyVista picks a backend: this script must run without a display.
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
import pyvista as pv  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))  # repo root

import nibabel as nib  # noqa: E402

from morphometry.cartilage.knee import (  # noqa: E402
    FEMUR_SUBREGIONS, TIBIA_SUBREGIONS, Femur, Tibia, find_thinnest_area,
    find_thinnest_area_by_compartment, plot_knee_segments, plot_knee_thickness,
    plot_thinnest_point, scan_qc_flags,
)
from morphometry.cartilage.femur_regions import FEMUR_REGION_KEYS  # noqa: E402
from morphometry.cartilage.surface import triangle_areas, triangle_centroids  # noqa: E402
from morphometry.cartilage.tibia_regions import (  # noqa: E402
    TIBIA_REGION_KEYS, tibia_display_label,
)
from morphometry.image_io import Segmentation  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

DEFAULT_DATA_DIR = Path(os.environ.get(
    "MORPH_DFG_KNEES_DIR",
    "/home/simon/Data/dfg_knees/monai_segmentations/labels/original"))
DEFAULT_OUT = REPO / "out" / "knee_zones"

#: The two clean reference scans of the same knee at 0 and 90 degrees of flexion.
DEFAULT_SCAN = "K1_0_unbelastet"
DEFAULT_FLEXION_SCAN = "K1_90_unbelastet"

#: Scans known to be broken / QC-flagged (CLAUDE.md "Coordinate system"); refused
#: unless --allow-flagged, because a figure drawn from one of them illustrates the
#: defect rather than the subdivision.
FLAGGED_SCANS = ("K7_90_belastet", "K8_90_belastet", "K11_90_belastet")

#: dfg_knees label map. Only the cartilage labels are figure arguments; the bone
#: labels come from the library's own defaults (see DEFAULTS below).
TIBIA_CARTILAGE_LABEL = 4
FEMUR_CARTILAGE_LABEL = 3

# The library's own defaults, read off the signatures: this script illustrates the
# measurement, so the subdivision it draws must be the subdivision the measurements use.
DEFAULTS = {
    "tibia_bone_label": inspect.signature(Tibia.__init__).parameters["bone_label"].default,
    "femur_bone_label": inspect.signature(Femur.__init__).parameters["bone_label"].default,
    "central_fraction": inspect.signature(Tibia.__init__).parameters["central_fraction"].default,
    "weightbearing_fraction":
        inspect.signature(Femur.__init__).parameters["weightbearing_fraction"].default,
}

# Overlay colours. Deliberately outside the region colour tables so an overlay can
# never be mistaken for a region.
OVERLAY = {
    "ellipse": "#111111",      # the central region's elliptical boundary
    "wedge": "#444444",        # the two +-45 degree wedge planes
    "internal": "#00429D",     # tibia: towards the other plate; femur: the ml axis
    "anterior": "#B8002E",     # the anterior direction (both bones)
    "proximal": "#111111",     # femur: the shaft axis
    "notch": "#111111",        # femur: the trochlear notch point and its border plane
    "border": "#555555",       # femur: each condyle's 60% posterior border plane
    "denuded": "#FF00FF",      # 0 mm (denuded / rim) tAB triangles
    "bone": "#C8C8C8",         # the rest of the bone surface
}

#: Canvas colour. Light, but not white: the library draws one frame arrow in white.
BACKGROUND = "#E9E9E9"

#: How much larger than the anatomy a frame is drawn, leaving room for the title block.
PADDING = 1.18

#: Turntable defaults (--animate). 48 frames at 16 fps is one 3 s revolution; the GIF is
#: written at half the PNG's linear size because a full-size 1600x1200 GIF of 48 frames
#: runs to tens of megabytes for no extra legibility (the MP4, which is inter-frame
#: compressed, is written at full size).
DEFAULT_FRAMES = 48
DEFAULT_FPS = 16
GIF_SCALE = 0.5

#: Set once by :func:`render` when ``--animate`` is given, read by :func:`_export`.
#: Module state rather than a parameter threaded through every ``figure_*`` function:
#: a turntable is a property of how a finished scene is *captured*, not of how it is
#: built, and every scene is captured through :func:`_export`.
_ANIMATION = None

# The library's own region -> colour tables, as lookups. Using these (rather than a copy)
# is what guarantees a legend entry in these figures marks the region it claims to.
TIBIA_COLORS = {attr: color for attr, _, color in TIBIA_SUBREGIONS}
FEMUR_COLORS = {attr: color for attr, _, color, _ in FEMUR_SUBREGIONS}

# World (RAS) anatomical directions. The cartilage meshes are mapped to RAS for both
# backends (see morphometry/cartilage/surface.py::index_to_world_affine), so these are
# anatomical, not array, directions.
WORLD_RIGHT = np.array([1.0, 0.0, 0.0])
WORLD_ANTERIOR = np.array([0.0, 1.0, 0.0])
WORLD_SUPERIOR = np.array([0.0, 0.0, 1.0])


# ----------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------

def load_scan(data_dir: Path, scan: str, allow_flagged: bool = False) -> Segmentation:
    """
    Load one ``dfg_knees`` segmentation and put it in LPI orientation.

    :param data_dir: Directory holding ``<scan>.nii.gz``.
    :param scan: Scan id without suffix, e.g. ``'K1_0_unbelastet'``.
    :param allow_flagged: Permit one of :data:`FLAGGED_SCANS` (whose segmentation is
        known to be broken or not to reorient to LPI).
    :return: The reoriented :class:`~morphometry.image_io.Segmentation`.
    :raises SystemExit: If the file is missing, or the scan is flagged and
        ``allow_flagged`` is False.
    """
    if scan in FLAGGED_SCANS and not allow_flagged:
        raise SystemExit(f"{scan} is QC-flagged (see CLAUDE.md); pass --allow-flagged to "
                         f"render it anyway.")
    path = Path(data_dir) / f"{scan}.nii.gz"
    if not path.exists():
        raise SystemExit(f"No segmentation at {path} (set --data-dir / MORPH_DFG_KNEES_DIR).")
    seg = Segmentation.from_nibabel(nib.load(str(path)))
    seg.transform_coordinate_system()
    return seg


def build_bones(seg: Segmentation, laterality: str = None) -> tuple:
    """
    Build and subdivide the ``Tibia`` and ``Femur`` of one knee scan.

    Both subdivisions are computed eagerly (``calculate_landmarks``) so the caller can
    report the region shares before any rendering happens.

    :param seg: The LPI-oriented knee segmentation.
    :param laterality: ``'left'`` / ``'right'`` knee, or ``None``. Only adds the paper's
        medial/lateral names; it is an *assumption* for ``dfg_knees`` (plan open issue 3).
    :return: ``(tibia, femur)``, both with their ``subdivision`` populated.
    """
    tibia = Tibia(seg, cartilage_label=TIBIA_CARTILAGE_LABEL,
                  bone_label=DEFAULTS["tibia_bone_label"], laterality=laterality)
    femur = Femur(seg, cartilage_label=FEMUR_CARTILAGE_LABEL,
                  bone_label=DEFAULTS["femur_bone_label"], laterality=laterality)
    tibia.calculate_landmarks()
    femur.calculate_landmarks()
    return tibia, femur


# ----------------------------------------------------------------------------------------
# Scene plumbing
# ----------------------------------------------------------------------------------------

def _orthonormal_up(view_dir: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    The component of ``up`` perpendicular to ``view_dir``, normalised.

    A camera up vector that is not perpendicular to the view direction leaves the
    in-plane rotation of the figure to VTK's own fix-up; doing it here keeps "anterior
    up" meaning the anterior direction projected into the image plane.

    :param view_dir: Unit vector from the subject towards the camera.
    :param up: The direction to put at the top of the frame.
    :return: A unit vector perpendicular to ``view_dir``.
    """
    view_dir = np.asarray(view_dir, dtype=float)
    up = np.asarray(up, dtype=float) - float(np.dot(up, view_dir)) * view_dir
    norm = float(np.linalg.norm(up))
    if norm < 1e-9:  # up parallel to the view: any perpendicular will do
        up = np.cross(view_dir, WORLD_ANTERIOR)
        norm = float(np.linalg.norm(up))
        if norm < 1e-9:
            up = np.cross(view_dir, WORLD_RIGHT)
            norm = float(np.linalg.norm(up))
    return up / norm


def _fit_scale(points: np.ndarray, view_dir: np.ndarray, up: np.ndarray,
               aspect: float) -> float:
    """
    The parallel-projection half-height (mm) that just contains ``points`` in one panel.

    :param points: ``(N, 3)`` world-mm points.
    :param view_dir: Unit vector from the anatomy towards the camera.
    :param up: Direction at the top of the frame (orthogonalised here).
    :param aspect: The panel's height / width ratio.
    :return: Half the frame height in mm, before any padding or zoom.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    view_dir = np.asarray(view_dir, dtype=float)
    view_dir = view_dir / np.linalg.norm(view_dir)
    up = _orthonormal_up(view_dir, up)
    right = np.cross(-view_dir, up)
    relative = points - points.mean(axis=0)
    return max(float(np.abs(relative @ up).max()),
               float(np.abs(relative @ right).max()) * aspect, 1e-3)


def _look_at(plotter: pv.Plotter, points: np.ndarray, view_dir: np.ndarray,
             up: np.ndarray, zoom: float = 1.0, scale: float = None) -> None:
    """
    Point the active subplot's camera at a point cloud from a given direction.

    The direction and up vector are set from the anatomy rather than from hard-coded
    azimuth / elevation angles, and the frame is fitted to ``points`` alone rather than
    to the scene: the bone meshes carry a long stretch of shaft that is invisible in
    these views (it extends along the camera axis) but would otherwise shrink the joint
    into a corner of the image.

    :param plotter: The ``Plotter`` (its currently active subplot is used).
    :param points: ``(N, 3)`` world-mm points to centre on and fit the frame to.
    :param view_dir: Unit vector from the anatomy towards the camera.
    :param up: Direction to put at the top of the frame (orthogonalised here).
    :param zoom: Extra zoom factor on top of the fit (>1 crops tighter).
    :param scale: A frame half-height in mm (see :func:`_fit_scale`) to use instead of
        fitting ``points``. Panels that are to be compared pass the same value -- each
        fitted in its *own* frame and the largest kept -- so that two scans of one knee
        are drawn at one magnification and a difference in the image is a difference in
        the anatomy.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    focal = points.mean(axis=0)
    span = max(float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))), 1.0)
    view_dir = np.asarray(view_dir, dtype=float)
    view_dir = view_dir / np.linalg.norm(view_dir)
    up = _orthonormal_up(view_dir, up)
    plotter.camera_position = [tuple(focal + 2.0 * span * view_dir), tuple(focal), tuple(up)]
    plotter.camera.focal_point = tuple(focal)

    # Fit by hand rather than with reset_camera: under parallel projection that sets the
    # scale from the bounding *sphere*, which for a flat, wide object (a tAB) leaves it
    # at half the panel height. PADDING leaves room for the title block.
    if scale is None:
        panel_w, panel_h = _panel_size_px(plotter)
        scale = _fit_scale(points, view_dir, up, panel_h / panel_w)
    plotter.camera.parallel_scale = (PADDING / zoom) * scale

    # Remember what this panel was pointed at, so a turntable can spin it about its own
    # anatomical up vector (see _turntable). Recording it here rather than re-deriving it
    # later is what lets every figure animate without its builder knowing about animation.
    cameras = getattr(plotter, "_zone_cameras", None)
    if cameras is None:
        cameras = plotter._zone_cameras = {}
    cameras[plotter.renderers.active_index] = {
        "focal": focal, "view_dir": view_dir, "up": up, "distance": 2.0 * span,
        "scale": float(plotter.camera.parallel_scale),
    }


def _title(plotter: pv.Plotter, text: str) -> None:
    """
    Write a figure / panel title into the active subplot's upper-left corner.

    :param plotter: The ``Plotter`` (its currently active subplot is used).
    :param text: The title; newlines are allowed and are rendered as further lines.
    """
    plotter.add_text(text, position="upper_left", font_size=10, color="black")


def _add_world_axes(plotter: pv.Plotter) -> None:
    """
    Add the RAS orientation marker to the active subplot (R = patient right, A, S).

    The meshes are in world mm, RAS, for both image backends, so the marker states the
    *anatomical* directions of the scene. It is what lets a reader check that the plate
    labelled ``cLT`` really is on the patient's left (smaller world x), which is how
    :func:`morphometry.cartilage.tibia_regions.find_plates` names the plates.

    :param plotter: The ``Plotter`` (its currently active subplot is used).
    """
    plotter.add_axes(xlabel="R", ylabel="A", zlabel="S", color="black",
                     labels_off=False, line_width=3)


def _rotate_about(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate ``vector`` about a unit ``axis`` by ``angle`` radians (Rodrigues' formula).

    :param vector: The vector to rotate.
    :param axis: A unit rotation axis.
    :param angle: The rotation angle in radians, right-handed about ``axis``.
    :return: The rotated vector.
    """
    vector = np.asarray(vector, dtype=float)
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    return (vector * np.cos(angle)
            + np.cross(axis, vector) * np.sin(angle)
            + axis * float(np.dot(axis, vector)) * (1.0 - np.cos(angle)))


def _write_movie(frames: list, stem: Path, fps: int, written: list = None) -> None:
    """
    Write a frame sequence as an animated GIF and, where ffmpeg exists, as an MP4.

    The GIF always works (pure Python, through imageio's Pillow plugin) but is large and
    palettised; the MP4 is inter-frame compressed and far smaller at full resolution, so
    both are written when they can be. ffmpeg is called as a subprocess rather than
    through ``imageio-ffmpeg``, which is not in ``requirements.txt``.

    :param frames: The captured frames as ``(H, W, 3)`` uint8 arrays.
    :param stem: Output path without a suffix; ``.gif`` / ``.mp4`` are appended.
    :param fps: Frames per second for both outputs.
    :param written: A list the written paths are appended to, for the final report.
    """
    gif = stem.with_suffix(".gif")
    small = [frame[::int(round(1 / GIF_SCALE)), ::int(round(1 / GIF_SCALE))] for frame in frames]
    imageio.mimsave(gif, small, duration=1000.0 / fps, loop=0)
    print(f"wrote {gif}")
    if written is not None:
        written.append(gif)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("  (no ffmpeg on PATH: GIF only)")
        return
    mp4 = stem.with_suffix(".mp4")
    with tempfile.TemporaryDirectory() as tmp:
        for i, frame in enumerate(frames):
            imageio.imwrite(Path(tmp) / f"frame_{i:04d}.png", frame)
        # yuv420p for players that refuse anything else; libx264 needs even dimensions,
        # and the panel sizes here are not guaranteed to be.
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps),
             "-i", str(Path(tmp) / "frame_%04d.png"),
             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", str(mp4)],
            check=True)
    print(f"wrote {mp4}")
    if written is not None:
        written.append(mp4)


def _turntable(plotter: pv.Plotter, stem: Path, frames: int, fps: int,
               written: list = None) -> None:
    """
    Spin every panel of a finished scene about its own anatomical up vector and film it.

    Each panel is rotated about the up vector :func:`_look_at` gave it, not about a
    single world axis: the panels of a comparison figure are posed in their own bone
    frames (two scans of one knee sit quite differently in the scanner), so a shared
    world axis would swing them out of step and destroy the very comparison the figure
    makes. The focal point, distance and parallel scale are left alone, so the anatomy
    neither drifts nor changes size as it turns.

    :param plotter: An off-screen ``Plotter`` whose scene is fully built, and whose
        panels were pointed with :func:`_look_at` (which records what it did).
    :param stem: Output path without a suffix (``.gif`` / ``.mp4`` are appended).
    :param frames: Number of frames in one full revolution.
    :param fps: Playback rate.
    :param written: A list the written paths are appended to, for the final report.
    """
    cameras = getattr(plotter, "_zone_cameras", None)
    if not cameras:  # a scene nobody pointed with _look_at: nothing sensible to spin
        print(f"  (no recorded camera: {stem.name} not animated)")
        return

    captured = []
    for step in range(frames):
        angle = 2.0 * np.pi * step / frames
        for index, camera in cameras.items():
            plotter.subplot(*plotter.renderers.index_to_loc(index))
            direction = _rotate_about(camera["view_dir"], camera["up"], angle)
            plotter.camera_position = [
                tuple(camera["focal"] + camera["distance"] * direction),
                tuple(camera["focal"]),
                tuple(camera["up"]),
            ]
            # camera_position resets the parallel scale, so restore the fitted one.
            plotter.camera.parallel_scale = camera["scale"]
        # render() explicitly: an off-screen screenshot otherwise hands back the image
        # from the last render, so moving the camera alone would film one still frame
        # over and over (and a GIF writer then collapses the lot into a single frame).
        # Copy it too -- the returned array is the plotter's own buffer, reused per call.
        plotter.render()
        captured.append(np.array(plotter.screenshot(return_img=True), copy=True))
    _write_movie(captured, stem, fps, written)


def _export(plotter: pv.Plotter, png: Path, html: Path = None,
            written: list = None) -> None:
    """
    Write a finished off-screen scene as a PNG and optionally as interactive HTML.

    :param plotter: The off-screen ``Plotter`` to capture (closed afterwards).
    :param png: Where to write the screenshot.
    :param html: Where to write a self-contained, rotatable HTML, or ``None`` to skip
        it (multi-panel scenes: PyVista's exporter only carries one renderer).
    :param written: A list the written paths are appended to, for the final report.
    """
    png.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(png))
    print(f"wrote {png}")
    if written is not None:
        written.append(png)
    if html is not None:
        plotter.export_html(str(html))
        print(f"wrote {html}")
        if written is not None:
            written.append(html)
    if _ANIMATION is not None:
        _turntable(plotter, png.with_suffix(""), _ANIMATION["frames"], _ANIMATION["fps"],
                   written)
    plotter.close()


def _panel_html(draw, scans: list, scale: float, size: tuple, html: bool,
                out_dir: Path, stem: str, written: list) -> None:
    """
    Re-draw each panel of a comparison figure on its own, as rotatable HTML.

    PyVista's HTML exporter carries a single renderer, so a multi-panel figure cannot be
    exported as one interactive scene. Each panel is therefore rebuilt in its own
    one-panel plotter and exported separately, and a small frameset page puts the two
    side by side so the comparison survives into the browser. The panels keep the shared
    ``scale`` of the PNG, so the two scenes are still drawn at one magnification.

    :param draw: ``draw(plotter, scan, scale)``, the panel builder of the calling figure.
    :param scans: The scan ids, one panel each, in the order they should appear.
    :param scale: The frame half-height in mm shared by the panels (see :func:`_look_at`).
    :param size: ``(width, height)`` of the *whole* figure in pixels; each panel gets an
        equal share of the width.
    :param html: Whether HTML was asked for at all; a no-op when false.
    :param out_dir: Directory the files are written to.
    :param stem: File name stem of the parent figure; panels get ``_<scan>`` appended.
    :param written: A list the written paths are appended to, for the final report.
    """
    if not html:
        return
    panel_size = (max(size[0] // len(scans), 400), size[1])
    pages = []
    for scan in scans:
        # Exported directly rather than through _export: the parent figure is already the
        # still, and filming a turntable of each panel as well would double the run for a
        # view the combined page lets the reader turn by hand anyway.
        plotter = _plotter(panel_size)
        draw(plotter, scan, scale)
        page = out_dir / f"{stem}_{scan}.html"
        plotter.export_html(str(page))
        plotter.close()
        print(f"wrote {page}")
        written.append(page)
        pages.append(page)
    _write_comparison_page(pages, scans, out_dir / f"{stem}.html", written)


def _write_comparison_page(pages: list, titles: list, path: Path,
                           written: list = None) -> None:
    """
    Write a page that embeds several exported scenes side by side in iframes.

    :param pages: The exported single-scene HTML files, in display order.
    :param titles: One caption per page.
    :param path: Where to write the combined page.
    :param written: A list the written path is appended to, for the final report.
    """
    columns = "".join(
        f'<figure><figcaption>{title}</figcaption>'
        f'<iframe src="{page.name}" loading="lazy"></iframe></figure>'
        for page, title in zip(pages, titles))
    path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        f"<title>{path.stem}</title>"
        "<style>body{font:14px/1.4 system-ui,sans-serif;margin:1rem;background:#e9e9e9}"
        "div{display:flex;gap:1rem;flex-wrap:wrap}figure{flex:1 1 420px;margin:0}"
        "figcaption{font-weight:600;margin-bottom:.4rem}"
        "iframe{width:100%;aspect-ratio:4/3;border:1px solid #bbb;background:#fff}</style>"
        f"<h1>{path.stem}</h1><p>Each panel rotates independently.</p><div>{columns}</div>",
        encoding="utf-8")
    print(f"wrote {path}")
    if written is not None:
        written.append(path)


def _plotter(size: tuple, shape: tuple = (1, 1)) -> pv.Plotter:
    """
    A light-background off-screen ``Plotter`` of the requested panel layout.

    The background is a very light grey rather than white on purpose: the bone surfaces,
    the white backing plates of the region labels and (if ``plot_regions`` is ever asked
    for its own frames, which draws one arrow white) the overlays all have white or
    near-white parts that vanish on a white canvas.

    :param size: ``(width, height)`` of the whole figure in pixels.
    :param shape: ``(rows, columns)`` of subplots.
    :return: The ``Plotter``, never shown (the caller passes it to :func:`_export`).
    """
    pv.set_plot_theme("document")
    plotter = pv.Plotter(off_screen=True, shape=shape, window_size=list(size))
    plotter.set_background(BACKGROUND)
    # Parallel projection: an anatomical figure should not foreshorten, and it lets the
    # overlays be offset towards the camera purely to avoid occlusion, without moving or
    # growing in the image the way perspective would. Per *renderer*: the Plotter method
    # only touches the active subplot, which would leave every other panel on perspective
    # and silently ignoring the frame scale set in _look_at.
    for renderer in plotter.renderers:
        renderer.enable_parallel_projection()
    return plotter


def _legend_entries(subdivision, keys, colors, display=lambda key: key) -> list:
    """
    ``[[label, colour], ...]`` for the regions a subdivision actually has.

    The colours come from the library's own ``TIBIA_SUBREGIONS`` / ``FEMUR_SUBREGIONS``
    tables, which is what ``plot_regions`` paints the mesh with, so a legend entry cannot
    disagree with the region it marks. With a known laterality the paper's medial/lateral
    name is appended in brackets.

    :param subdivision: A ``TibiaSubdivision`` / ``FemurSubdivision``.
    :param keys: The ordered region keys (region code = index into this sequence).
    :param colors: ``{key: colour}``.
    :param display: Maps a region key to its display label.
    :return: A list of ``[label, colour]`` pairs, in region-code order.
    """
    names = subdivision.anatomical_names
    present = set(np.asarray(subdivision.mesh.cell_data["region"]).tolist())
    return [[display(key) + (f" ({names[key]})" if names else ""), colors[key]]
            for code, key in enumerate(keys) if code in present]


def _both_legend(plotter: pv.Plotter, tibia_subdivision, femur_subdivision) -> None:
    """
    One legend for a scene holding both bones' subdivisions.

    Drawing both bones adds two legends on top of each other (each ``plot_regions`` call
    adds its own); this replaces them with a single box listing all twenty regions.

    :param plotter: The ``Plotter`` (its currently active subplot is used).
    :param tibia_subdivision: The ``TibiaSubdivision`` in the scene.
    :param femur_subdivision: The ``FemurSubdivision`` in the scene.
    """
    entries = (_legend_entries(tibia_subdivision, TIBIA_REGION_KEYS, TIBIA_COLORS,
                               tibia_display_label)
               + _legend_entries(femur_subdivision, FEMUR_REGION_KEYS, FEMUR_COLORS))
    _relegend(plotter, entries)


def _panel_size_px(plotter: pv.Plotter) -> tuple:
    """
    The pixel size of the active subplot.

    PyVista sizes 2D actors (legends, scalar bars) in fractions of their own renderer, so
    a box that is readable in a full-window scene is unreadable in a half-window panel and
    vice versa. Knowing the panel's pixel size lets the legend be asked for in pixels.

    :param plotter: The ``Plotter`` (its currently active subplot is used).
    :return: ``(width_px, height_px)`` of the active renderer's viewport.
    """
    x0, y0, x1, y1 = plotter.renderer.GetViewport()
    width, height = plotter.window_size
    return max((x1 - x0) * width, 1.0), max((y1 - y0) * height, 1.0)


def _relegend(plotter: pv.Plotter, entries: list, entry_px: float = 26.0) -> None:
    """
    Replace the legend ``plot_regions`` drew with a readable, correctly placed one.

    ``Tibia.plot_regions`` / ``Femur.plot_regions`` call ``add_legend()`` with PyVista's
    default size, which is a fraction of the *panel*: in a multi-panel figure that comes
    out unreadably small, and in a scene holding both bones there are two boxes on top of
    each other. The entries are rebuilt from the same colour tables, so nothing about the
    mapping changes -- only the box. Its size is requested in pixels (VTK scales the font
    to the box, so a fixed fraction of the panel gives a font that depends on the layout
    and on the number of entries) and its position is set by hand, because PyVista 0.44's
    ``map_loc_to_pos`` computes the horizontal offset of ``loc='upper right'`` from the
    legend's *height* (``x = 1 - size[1] - border``), which lands a tall legend in the
    middle of the panel.

    :param plotter: The ``Plotter`` (its currently active subplot is used).
    :param entries: ``[[label, colour], ...]`` from :func:`_legend_entries`.
    :param entry_px: Height of one legend row in pixels.
    """
    if not entries:
        return
    panel_w, panel_h = _panel_size_px(plotter)
    longest = max(len(str(label)) for label, _ in entries)
    width = min((0.62 * entry_px * longest + 1.6 * entry_px) / panel_w, 0.45)
    height = min(entry_px * len(entries) / panel_h, 0.92)
    try:
        plotter.remove_legend(render=False)
    except Exception:  # no legend on this renderer; nothing to replace
        pass
    legend = plotter.add_legend(labels=entries, bcolor="white", size=(width, height),
                                face="circle")
    legend.SetPosition(1.0 - width - 0.02, 0.97 - height)


# ----------------------------------------------------------------------------------------
# Overlays: the construction geometry the subdivision is cut with
# ----------------------------------------------------------------------------------------

def _region_centroids(subdivision, keys) -> dict:
    """
    The area-weighted centroid of every region of a subdivision's labelled tAB mesh.

    Used to place one text label per region, so that a reader can check the legend
    colour against the region it actually marks.

    :param subdivision: A ``TibiaSubdivision`` / ``FemurSubdivision``.
    :param keys: The ordered region keys (region code = index into this sequence).
    :return: ``{key: (3,) world-mm centroid}``, omitting regions with no triangles.
    """
    mesh = subdivision.mesh
    codes = np.asarray(mesh.cell_data["region"])
    centroids = triangle_centroids(mesh)
    areas = triangle_areas(mesh)
    out = {}
    for code, key in enumerate(keys):
        selected = codes == code
        if not selected.any():
            continue
        weight = areas[selected]
        out[key] = (centroids[selected] * weight[:, None]).sum(axis=0) / weight.sum()
    return out


def _add_region_labels(plotter: pv.Plotter, subdivision, keys, lift: np.ndarray,
                       lift_mm: float, display=lambda key: key) -> None:
    """
    Label every region of a labelled tAB mesh at its own centroid.

    :param plotter: The ``Plotter`` (its currently active subplot is used).
    :param subdivision: A ``TibiaSubdivision`` / ``FemurSubdivision``.
    :param keys: The ordered region keys (region code = index into this sequence).
    :param lift: Unit direction the labels are offset along, towards the camera, so they
        are not buried in the surface.
    :param lift_mm: How far along ``lift`` to offset them.
    :param display: Maps a region key to the text drawn for it.
    """
    centroids = _region_centroids(subdivision, keys)
    if not centroids:
        return
    points = np.array([c + lift_mm * lift for c in centroids.values()])
    plotter.add_point_labels(points, [display(k) for k in centroids], font_size=13,
                             text_color="black", shape_color="white", shape_opacity=0.65,
                             point_size=1, always_visible=True)


def _plate_extent(plate_mesh_points: np.ndarray, centroid: np.ndarray) -> float:
    """
    Largest distance from a plate's centre of gravity to one of its points.

    Sets the size of the drawn wedge planes and direction arrows so they span the plate
    without dwarfing it.

    :param plate_mesh_points: ``(N, 3)`` world-mm points of the plate.
    :param centroid: The plate's centre of gravity S.
    :return: The maximum radius in mm (at least 1 mm).
    """
    if len(plate_mesh_points) == 0:
        return 1.0
    return max(float(np.linalg.norm(plate_mesh_points - centroid, axis=1).max()), 1.0)


def add_tibia_construction(plotter: pv.Plotter, subdivision, lift_mm: float = 2.0) -> None:
    """
    Draw the geometry each tibial plate's five regions are cut with.

    Per plate (both patient sides), in its own ``PlateFrame``:

    * the **central ellipse**: ``rho = ellipse_level`` in the plate plane, i.e. the
      boundary of the central region. The central region is an elliptical *cylinder*
      along the plate normal ``b3``, so lifting the drawn ring along ``b3`` (for
      visibility above the curved surface) does not distort it -- it projects exactly
      onto the boundary the mesh was cut at.
    * the two **+-45 degree wedge planes** through S, with normals ``ml - ap`` and
      ``ml + ap``: exactly the planes
      :func:`~morphometry.cartilage.tibia_regions._subdivide_plate` splits the
      peripheral ring with. Seen down ``b3`` they appear as the two diagonal lines of
      the paper's Fig. 5A.
    * the **internal** direction (towards the other plate, ``tc`` projected into the
      plate plane) and the **anterior** direction, as arrows from S, plus a marker at S.

    :param plotter: The ``Plotter`` (its currently active subplot is used).
    :param subdivision: The ``TibiaSubdivision`` to draw the construction of.
    :param lift_mm: How far above the plate (along ``b3``) the ellipse ring is drawn.
    """
    mesh = subdivision.mesh
    plate_ids = np.asarray(mesh.cell_data["plate"])
    angles = np.linspace(0.0, 2.0 * np.pi, 241)
    for index, (side, frame) in enumerate(sorted(subdivision.plates.items())):
        b1, b2, b3 = frame.axes
        semi_a, semi_b = (float(v) for v in frame.semi_axes)
        ring = (frame.centroid + lift_mm * b3
                + np.cos(angles)[:, None] * (semi_a * b1)
                + np.sin(angles)[:, None] * (semi_b * b2))
        plotter.add_mesh(pv.MultipleLines(ring).tube(radius=0.35),
                         color=OVERLAY["ellipse"])

        # Size the planes / arrows to this plate, not to the whole scene.
        plate_cells = np.flatnonzero(plate_ids == index)
        points = (np.asarray(mesh.extract_cells(plate_cells).points, dtype=float)
                  if len(plate_cells) else np.empty((0, 3)))
        extent = _plate_extent(points, frame.centroid)
        for normal in (frame.ml_direction - frame.anterior_direction,
                       frame.ml_direction + frame.anterior_direction):
            plotter.add_mesh(pv.Plane(center=frame.centroid, direction=normal,
                                      i_size=1.6 * extent, j_size=1.6 * extent),
                             color=OVERLAY["wedge"], opacity=0.18)
        plotter.add_mesh(pv.Sphere(radius=0.9, center=frame.centroid), color="black")
        for direction, key in ((frame.ml_direction, "internal"),
                               (frame.anterior_direction, "anterior")):
            plotter.add_mesh(pv.Arrow(frame.centroid + lift_mm * b3, direction,
                                      scale=0.9 * extent, tip_length=0.18,
                                      tip_radius=0.055, shaft_radius=0.018),
                             color=OVERLAY[key])


def _rectangle_outline(center: np.ndarray, u: np.ndarray, v: np.ndarray,
                       half_u: float, half_v: float, radius: float = 0.5) -> pv.PolyData:
    """
    The outline of a rectangle in the plane spanned by ``u`` and ``v``, as a tube.

    Used instead of a filled ``pv.Plane`` for the femoral border planes: a filled plane
    is informative seen edge-on but hides the whole bone when seen face-on, whereas an
    outline marks the plane's position from every direction (and still projects onto a
    single line when seen edge-on).

    :param center: The rectangle's centre in world mm.
    :param u: Unit in-plane direction of the first side.
    :param v: Unit in-plane direction of the second side.
    :param half_u: Half the side length along ``u`` (mm).
    :param half_v: Half the side length along ``v`` (mm).
    :param radius: Tube radius of the drawn outline (mm).
    :return: The outline as a tube mesh.
    """
    corners = np.array([center + su * half_u * u + sv * half_v * v
                        for su, sv in ((1, 1), (-1, 1), (-1, -1), (1, -1), (1, 1))])
    return pv.MultipleLines(corners).tube(radius=radius)


def add_femur_construction(plotter: pv.Plotter, subdivision) -> None:
    """
    Draw the femoral frame and the borders the condylar regions are cut at.

    Everything is read off the ``FemoralFrame`` / ``CondyleFrame`` the subdivision was
    built with, so the drawn construction is the one the regions were actually cut with:

    * the **trochlear-notch point** (the frame's origin landmark) as a black sphere;
    * the frame's three axes as arrows -- ``ml`` (blue, towards the patient's right),
      ``ap`` (red, anterior) and ``proximal`` (black, up the shaft). They are drawn from
      a point offset distally from the notch so they are not buried inside the condyles,
      with a thin stalk back to the notch point itself.
    * the **notch plane** (black outline), the anterior border of both weight-bearing
      condyles, and each condyle's **60% posterior border** (grey outline), at
      ``notch_ap - weightbearing_fraction * (notch_ap - posterior_ap)``. They are drawn
      as outlines rather than filled planes: a filled plane seen face-on (the posterior
      view) covers the bone it is meant to annotate.

    :param plotter: The ``Plotter`` (its currently active subplot is used).
    :param subdivision: The ``FemurSubdivision`` to draw the construction of.
    """
    frame = subdivision.frame
    points = np.asarray(subdivision.mesh.points, dtype=float)
    width = float(np.ptp(points @ frame.ml_axis))
    height = float(np.ptp(points @ frame.proximal_axis))
    centre = points.mean(axis=0)

    for level, color in ([(frame.notch_ap, OVERLAY["notch"])]
                         + [(c.boundary_ap, OVERLAY["border"])
                            for c in subdivision.condyles.values()]):
        origin = centre + (level - float(centre @ frame.ap_axis)) * frame.ap_axis
        plotter.add_mesh(_rectangle_outline(origin, frame.ml_axis, frame.proximal_axis,
                                            0.55 * width, 0.55 * height, radius=0.35),
                         color=color)

    # Distal of the condyles, i.e. towards the camera in the inferior view (parallel
    # projection, so the offset costs nothing but the occlusion it avoids).
    origin = frame.notch_point - 0.75 * height * frame.proximal_axis
    plotter.add_mesh(pv.Sphere(radius=1.4, center=frame.notch_point), color=OVERLAY["notch"])
    plotter.add_mesh(pv.MultipleLines(np.array([frame.notch_point, origin])).tube(radius=0.3),
                     color=OVERLAY["notch"])
    for direction, key in ((frame.ml_axis, "internal"), (frame.ap_axis, "anterior"),
                           (frame.proximal_axis, "proximal")):
        plotter.add_mesh(pv.Arrow(origin, direction, scale=0.4 * width, tip_length=0.18,
                                  tip_radius=0.055, shaft_radius=0.018),
                         color=OVERLAY[key])
    plotter.add_point_labels(
        np.array([origin + 0.44 * width * frame.ml_axis,
                  origin + 0.44 * width * frame.ap_axis,
                  origin + 0.44 * width * frame.proximal_axis,
                  frame.notch_point]),
        ["ml (patient right)", "ap (anterior)", "proximal", "notch"],
        font_size=12, text_color="black", shape_color="white", shape_opacity=0.7,
        point_size=1, always_visible=True)


def _plate_view(subdivision) -> tuple:
    """
    A superior, anterior-up camera built from the plates' own frames.

    Averaging the two plates' normals ``b3`` (oriented superior) and anterior directions
    makes the view follow the tibial plateau rather than the scanner, which is what keeps
    the 0 and 90 degree flexion panels comparable.

    :param subdivision: The ``TibiaSubdivision``.
    :return: ``(view_dir, up)``, both unit vectors in world mm.
    """
    normals = np.array([frame.axes[2] for frame in subdivision.plates.values()])
    anterior = np.array([frame.anterior_direction for frame in subdivision.plates.values()])
    view_dir = normals.mean(axis=0)
    view_dir = view_dir / np.linalg.norm(view_dir)
    return view_dir, _orthonormal_up(view_dir, anterior.mean(axis=0))


def _femur_view(subdivision, view: str) -> tuple:
    """
    An inferior or posterior camera built from the femur's own frame.

    Using the ``FemoralFrame`` (shaft axis, posterior condylar line) rather than world
    axes is what makes the 0 and 90 degree panels show the same aspect of the bone.

    :param subdivision: The ``FemurSubdivision``.
    :param view: ``'inferior'`` (down the shaft axis onto the condyles, anterior up,
        the paper's Fig. 6A) or ``'posterior'`` (from behind, proximal up).
    :return: ``(view_dir, up)``, both unit vectors in world mm.
    """
    frame = subdivision.frame
    if view == "inferior":
        return -frame.proximal_axis, frame.ap_axis
    if view == "posterior":
        return -frame.ap_axis, frame.proximal_axis
    raise ValueError(f"unknown femoral view {view!r}")


# ----------------------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------------------

def report_shares(scan: str, tibia: Tibia, femur: Femur) -> dict:
    """
    Print the numbers the figures should be checked against, and return them.

    These are the quantities the subdivision is *defined* by, so a figure that does not
    match them is wrong: each tibial central region is ``central_fraction`` (0.2) of its
    plate's tAB by construction, each femoral condylar third is one third of its
    condyle's weight-bearing tAB, and the patient-left plate is the one with the smaller
    RAS world x (:func:`~morphometry.cartilage.tibia_regions.find_plates`).

    :param scan: The scan id, for the printed header.
    :param tibia: The subdivided ``Tibia``.
    :param femur: The subdivided ``Femur``.
    :return: ``{'tibia_shares', 'femur_shares', 'plate_x', 'qc_flags', 'transfer'}``.
    """
    tibia_shares = tibia.subdivision.region_shares
    femur_shares = femur.subdivision.region_shares
    plate_x = {side: float(frame.centroid[0])
               for side, frame in tibia.subdivision.plates.items()}
    flags = scan_qc_flags(tibia.image, tibia=tibia, femur=femur)

    print(f"\n--- {scan} ---")
    print("tibial region shares of their own plate's tAB (%):")
    for side in ("left", "right"):
        row = {tibia_display_label(k): round(100 * tibia_shares[k], 2)
               for k in TIBIA_REGION_KEYS if k[1] == side[0]}
        print(f"  patient-{side} plate: {row}  (central must be "
              f"{100 * DEFAULTS['central_fraction']:.0f}%)")
    print("femoral thirds' shares of their own weight-bearing condyle's tAB (%):")
    for side in ("left", "right"):
        row = {k: round(100 * femur_shares[k], 2)
               for k in FEMUR_REGION_KEYS if k in femur_shares and k[2] == side[0]}
        print(f"  patient-{side} condyle: {row}  (each must be ~33.33%)")
    print(f"plate centre-of-gravity world x (RAS, mm): {plate_x} -- "
          f"patient-left must be the smaller")
    print(f"plate split: {tibia.subdivision.plate_split}, dropped covered area "
          f"{tibia.subdivision.dropped_area:.2f} mm^2")
    print(f"region transfer (voxel -> tAB): tibia far_fraction "
          f"{tibia.transfer_qc['far_fraction']:.4f} max {tibia.transfer_qc['max_mm']:.2f} mm, "
          f"femur far_fraction {femur.transfer_qc['far_fraction']:.4f} "
          f"max {femur.transfer_qc['max_mm']:.2f} mm")
    print(f"scan_qc_flags: {flags or '(clean)'}")
    return {"tibia_shares": tibia_shares, "femur_shares": femur_shares,
            "plate_x": plate_x, "qc_flags": flags,
            "transfer": {"tibia": tibia.transfer_qc, "femur": femur.transfer_qc}}


def _laterality_note(laterality: str) -> str:
    """
    The title suffix stating how the region keys should be read.

    :param laterality: ``'left'`` / ``'right'`` or ``None``.
    :return: A short string for the figure titles.
    """
    if laterality is None:
        return "patient-side keys (laterality unrecorded; NOT medial/lateral)"
    return (f"patient-side keys; medial/lateral names from --laterality {laterality} "
            f"(ASSUMPTION, not recorded in the data)")


# ----------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------

def figure_tibia(tibia: Tibia, scan: str, out_dir: Path, size: tuple, html: bool,
                 written: list) -> None:
    """
    Figure 1: the tibial tAB subdivision from superior, with its construction geometry.

    The paper's Fig. 5A view: both plates seen down their own (superior) normal with
    anterior up, one colour per region (the library's ``TIBIA_SUBREGIONS`` table), the
    central ellipse, the two wedge planes, the internal / anterior directions and a text
    label per region. The central share of each plate is printed in the title, which is
    the number the drawn ellipse has to be consistent with.

    :param tibia: The subdivided ``Tibia``.
    :param scan: Scan id, used in the title and the file name.
    :param out_dir: Directory the files are written to.
    :param size: ``(width, height)`` in pixels.
    :param html: Also export interactive HTML.
    :param written: List the written paths are appended to.
    """
    sub = tibia.subdivision
    view_dir, up = _plate_view(sub)
    shares = sub.region_shares
    plotter = _plotter(size)
    tibia.plot_regions(plotter=plotter, show=False, show_bone=True, show_frames=False)
    add_tibia_construction(plotter, sub)
    _relegend(plotter, _legend_entries(sub, TIBIA_REGION_KEYS, TIBIA_COLORS,
                                      tibia_display_label))
    _add_region_labels(plotter, sub, TIBIA_REGION_KEYS, view_dir, 6.0,
                       display=tibia_display_label)
    _add_world_axes(plotter)
    _title(plotter, (
        f"Tibial subchondral bone area (tAB), subdivided — {scan}\n"
        f"viewed down the plates' own normal (superior), anterior up — "
        f"{_laterality_note(tibia.laterality)}\n"
        f"black ring = central ellipse (cLT {100 * shares['clt']:.1f}% / "
        f"cRT {100 * shares['crt']:.1f}% of its plate's tAB), grey planes = ±45° wedges, "
        f"blue = internal, red = anterior"))
    _look_at(plotter, sub.mesh.points, view_dir, up)
    _export(plotter, out_dir / f"tibia_regions_{scan}.png",
            out_dir / f"tibia_regions_{scan}.html" if html else None, written)


def figure_femur(femur: Femur, scan: str, out_dir: Path, size: tuple, html: bool,
                 written: list) -> None:
    """
    Figure 2: the femoral tAB subdivision from inferior and from posterior.

    The paper's Fig. 6A view plus a posterior one. ``Femur.plot_regions`` draws the
    femoral frame itself (the ``ml`` / ``ap`` / ``proximal`` axes as arrows at the
    trochlear notch, the notch point as a sphere, and the notch plane and each condyle's
    60% posterior border as translucent planes), so the construction geometry here is the
    library's own; this figure only frames it, labels the regions and names the axes.
    Those planes are cut perpendicular to the ``ap`` axis, so they are seen edge-on from
    inferior (where they read as the borders they are) but face-on from posterior (where
    they would cover the whole bone); the posterior panel therefore shows the regions
    alone, and the borders are read off the colour changes. The two panels are one
    figure; the interactive HTML is the inferior view alone, because PyVista's HTML
    exporter only carries a single renderer.

    :param femur: The subdivided ``Femur``.
    :param scan: Scan id, used in the titles and the file names.
    :param out_dir: Directory the files are written to.
    :param size: ``(width, height)`` in pixels.
    :param html: Also export interactive HTML of the inferior view.
    :param written: List the written paths are appended to.
    """
    sub = femur.subdivision
    shares = sub.region_shares
    thirds = ", ".join(f"{k} {100 * shares[k]:.1f}%" for k in sorted(shares))

    plotter = _plotter(size, shape=(1, 2))
    for column, view in enumerate(("inferior", "posterior")):
        plotter.subplot(0, column)
        femur.plot_regions(plotter=plotter, show=False, show_bone=True, show_frames=False)
        add_femur_construction(plotter, sub)
        _relegend(plotter, _legend_entries(sub, FEMUR_REGION_KEYS, FEMUR_COLORS))
        view_dir, up = _femur_view(sub, view)
        _add_region_labels(plotter, sub, FEMUR_REGION_KEYS, view_dir, 7.0)
        _add_world_axes(plotter)
        if view == "inferior":
            _title(plotter, (
                f"Femoral tAB, subdivided — {scan} — from inferior (paper Fig. 6A), "
                f"anterior up — {_laterality_note(femur.laterality)}\n"
                f"frame arrows: blue = ml (patient right), red = ap (anterior), black = "
                f"proximal; black sphere = the trochlear notch\n"
                f"outlines, seen edge-on: black = the notch plane (anterior border of the "
                f"weight-bearing condyle cF), grey = each condyle's "
                f"{100 * DEFAULTS['weightbearing_fraction']:.0f}% posterior border"))
        else:
            _title(plotter, (
                f"the same, from posterior, proximal up — the thirds (i/c/e) run across "
                f"each condyle, the zones (a.., p..) along it\n"
                f"the border outlines are now face-on: everything anterior of the black "
                f"rectangle is trochlea (alf / arf), everything\nposterior of a grey one "
                f"is a posterior-condyle zone (plf / prf), the three thirds lie between "
                f"them"))
        _look_at(plotter, sub.mesh.points, view_dir, up)
    _export(plotter, out_dir / f"femur_regions_{scan}.png", None, written)
    print(f"  femoral thirds of their own condyle's weight-bearing tAB: {thirds}")

    # Single-renderer repeat of the inferior view, for the rotatable HTML.
    if html:
        plotter = _plotter(size)
        femur.plot_regions(plotter=plotter, show=False, show_bone=True, show_frames=False)
        add_femur_construction(plotter, sub)
        _relegend(plotter, _legend_entries(sub, FEMUR_REGION_KEYS, FEMUR_COLORS))
        view_dir, up = _femur_view(sub, "inferior")
        _add_region_labels(plotter, sub, FEMUR_REGION_KEYS, view_dir, 7.0)
        _add_world_axes(plotter)
        _title(plotter, f"Femoral tAB, subdivided — {scan} — from inferior\n"
                        f"{_laterality_note(femur.laterality)}")
        _look_at(plotter, sub.mesh.points, view_dir, up)
        _export(plotter, out_dir / f"femur_regions_inferior_{scan}.png",
                out_dir / f"femur_regions_inferior_{scan}.html", written)


def figure_both(tibia: Tibia, femur: Femur, scan: str, out_dir: Path, size: tuple,
                html: bool, written: list) -> None:
    """
    Figure 3: both bones in one scene — the labelled tAB, and the voxels it was transferred to.

    Left panel: both subdivided tAB meshes (``plot_regions``), which is where the regions
    are *defined*. Right panel: the cartilage voxels of both bones coloured by the region
    they were given (``plot_knee_segments``), which is what the thickness statistics and
    the scripts' CSV rows are keyed by. Seen from anterior, superior up, so the
    femorotibial relationship is visible.

    :param tibia: The subdivided ``Tibia``.
    :param femur: The subdivided ``Femur`` of the same knee.
    :param scan: Scan id, used in the titles and the file names.
    :param out_dir: Directory the files are written to.
    :param size: ``(width, height)`` in pixels.
    :param html: Also export interactive HTML of the tAB panel.
    :param written: List the written paths are appended to.
    """
    points = np.vstack([np.asarray(tibia.subdivision.mesh.points, dtype=float),
                        np.asarray(femur.subdivision.mesh.points, dtype=float)])
    # Anterior, tilted towards the patient's right: a straight anterior view flattens the
    # joint into a silhouette, and a little obliquity shows the femur sitting on the plateau.
    view_dir = WORLD_ANTERIOR + 0.45 * WORLD_RIGHT
    up = WORLD_SUPERIOR

    plotter = _plotter(size, shape=(1, 2))
    plotter.subplot(0, 0)
    tibia.plot_regions(plotter=plotter, show=False, show_bone=True, show_frames=False)
    femur.plot_regions(plotter=plotter, show=False, show_bone=True, show_frames=False)
    _both_legend(plotter, tibia.subdivision, femur.subdivision)
    _add_world_axes(plotter)
    _title(plotter, f"Both bones' subdivided tAB — {scan}\n"
                    f"where the regions are defined — {_laterality_note(tibia.laterality)}")
    _look_at(plotter, points, view_dir, up, zoom=0.85)

    plotter.subplot(0, 1)
    plot_knee_segments(tibia, femur, plotter=plotter, show=False)
    _both_legend(plotter, tibia.subdivision, femur.subdivision)
    _add_world_axes(plotter)
    _title(plotter, f"The cartilage voxels those regions were transferred to — {scan}\n"
                    f"nearest tAB triangle per voxel; this is what the CSV rows count")
    _look_at(plotter, points, view_dir, up, zoom=0.85)
    plotter.link_views()
    _export(plotter, out_dir / f"knee_regions_and_segments_{scan}.png", None, written)

    if html:
        plotter = _plotter(size)
        tibia.plot_regions(plotter=plotter, show=False, show_bone=True, show_frames=False)
        femur.plot_regions(plotter=plotter, show=False, show_bone=True, show_frames=False)
        _both_legend(plotter, tibia.subdivision, femur.subdivision)
        _add_world_axes(plotter)
        _title(plotter, f"Both bones' subdivided tAB — {scan}\n"
                        f"{_laterality_note(tibia.laterality)}")
        _look_at(plotter, points, view_dir, up, zoom=0.85)
        _export(plotter, out_dir / f"knee_regions_{scan}.png",
                out_dir / f"knee_regions_{scan}.html", written)


def _en_face(subdivision, bone: str) -> tuple:
    """
    The camera that shows one bone's tAB face-on, from the joint space.

    :param subdivision: The bone's subdivision.
    :param bone: ``'tibia'`` (seen from superior, anterior up) or ``'femur'`` (from
        inferior, anterior up).
    :return: ``(view_dir, up)``, both unit vectors in world mm.
    """
    return _plate_view(subdivision) if bone == "tibia" else _femur_view(subdivision,
                                                                       "inferior")


def figure_thickness(tibia: Tibia, femur: Femur, scan: str, out_dir: Path, size: tuple,
                     html: bool, written: list) -> tuple:
    """
    Figure 4: the thickness heatmap on both tAB meshes, and the 0 mm band on its own.

    The first figure is the paper's Fig. 4B/5B: each bone's tAB seen face-on from the
    joint space (the tibia from superior, the femur from inferior) and coloured by the
    area-weighted surface thickness. Both panels share one colour scale, drawn once, and
    the limits are forced to start at 0 mm so the denuded triangles are *on* the scale
    rather than clipped off it. A straight anterior view of both bones at once was tried
    first and rejected: the femur hides the plateau and both tABs are seen edge-on, which
    is exactly where a heatmap carries no information. That whole-joint scene is still
    written, as the rotatable HTML, where the reader can turn it.

    Because 0 mm is still the dark end of a sequential colour map and easy to miss, the
    second figure paints exactly the 0 mm triangles magenta on an otherwise grey tAB, in
    the same two views, and prints their share.

    :param tibia: The subdivided ``Tibia``.
    :param femur: The subdivided ``Femur`` of the same knee.
    :param scan: Scan id, used in the titles and the file names.
    :param out_dir: Directory the files are written to.
    :param size: ``(width, height)`` in pixels.
    :param html: Also export interactive HTML of the whole-joint heatmap.
    :param written: List the written paths are appended to.
    :return: ``(tibia_thickness, femur_thickness)``, so the thinnest-point figure can
        reuse them instead of measuring again.
    """
    tibia_thickness = tibia.calculate_thickness()
    femur_thickness = femur.calculate_thickness(tibia=tibia)
    values = np.concatenate([np.asarray(tibia_thickness.triangle_thickness, dtype=float),
                             np.asarray(femur_thickness.triangle_thickness, dtype=float)])
    finite = values[np.isfinite(values)]
    clim = (0.0, float(np.nanmax(finite)))
    zero_share = float((finite == 0.0).mean())
    print(f"  thickness range {finite.min():.3f}-{finite.max():.3f} mm over "
          f"{len(finite)} tAB triangles; {100 * zero_share:.2f}% read exactly 0 mm "
          f"(denuded / rim)")

    panels = (("tibia", "Tibial", tibia, tibia_thickness, "from superior"),
              ("femur", "Femoral", femur, femur_thickness, "from inferior"))

    plotter = _plotter(size, shape=(1, 2))
    for column, (bone, name, obj, thickness, view) in enumerate(panels):
        plotter.subplot(0, column)
        obj.plot_thickness(thickness, plotter=plotter, show=False, clim=clim,
                           show_scalar_bar=(column == 0))
        _add_world_axes(plotter)
        _title(plotter, (
            f"{name} cartilage thickness on the tAB — {scan} — {view}, "
            f"anterior up\n"
            f"area-weighted surface distance (Wirth & Eckstein II-C); both panels share "
            f"the 0–{clim[1]:.2f} mm scale,\nforced to start at 0 so the denuded "
            f"triangles ({100 * zero_share:.2f}% of all tAB) are on it"))
        _look_at(plotter, obj.subdivision.mesh.points, *_en_face(obj.subdivision, bone))
    _export(plotter, out_dir / f"knee_thickness_{scan}.png", None, written)

    plotter = _plotter(size, shape=(1, 2))
    for column, (bone, name, obj, thickness, view) in enumerate(panels):
        plotter.subplot(0, column)
        mesh = obj.subdivision.mesh
        triangle = np.asarray(thickness.triangle_thickness, dtype=float)
        zero = np.flatnonzero(triangle == 0.0)
        nonzero = np.flatnonzero(triangle != 0.0)
        if len(nonzero):
            plotter.add_mesh(mesh.extract_cells(nonzero), color=OVERLAY["bone"])
        if len(zero):
            plotter.add_mesh(mesh.extract_cells(zero), color=OVERLAY["denuded"])
            # On a healthy knee the 0 mm triangles are a scatter of single faces, which
            # at this scale is a few pixels: mark each one with a point glyph as well,
            # so "almost none" is visible as almost none rather than as nothing.
            plotter.add_mesh(pv.PolyData(triangle_centroids(mesh)[zero]),
                             color=OVERLAY["denuded"], point_size=9,
                             render_points_as_spheres=True)
        _add_world_axes(plotter)
        share = float((triangle == 0.0).mean())
        _title(plotter, (
            f"Where the 0 mm thickness is — {bone} tAB, {scan}, {view}\n"
            f"magenta = tAB triangles with no cartilage above them (denuded bone, and "
            f"the rim of the coverage), each also marked\nwith a dot; grey = the rest "
            f"of the tAB. {len(zero)} of {len(triangle)} triangles = "
            f"{100 * share:.2f}% — this knee is essentially fully covered"))
        _look_at(plotter, mesh.points, *_en_face(obj.subdivision, bone))
    _export(plotter, out_dir / f"knee_thickness_denuded_{scan}.png", None, written)

    # The whole joint in one scene: edge-on in any single screenshot, but the view the
    # reader can actually turn, so it is the one exported as HTML.
    if html:
        points = np.vstack([np.asarray(tibia.subdivision.mesh.points, dtype=float),
                            np.asarray(femur.subdivision.mesh.points, dtype=float)])
        plotter = _plotter(size)
        plot_knee_thickness(tibia, femur, tibia_thickness, femur_thickness,
                            plotter=plotter, show=False, clim=clim)
        _add_world_axes(plotter)
        _title(plotter, f"Cartilage thickness on both tABs, one shared "
                        f"0–{clim[1]:.2f} mm scale — {scan}\nrotate to see either bone "
                        f"face-on")
        _look_at(plotter, points, WORLD_ANTERIOR + 0.45 * WORLD_RIGHT, WORLD_SUPERIOR)
        _export(plotter, out_dir / f"knee_thickness_joint_{scan}.png",
                out_dir / f"knee_thickness_joint_{scan}.html", written)
    return tibia_thickness, femur_thickness


def figure_flexion(scans: dict, out_dir: Path, size: tuple, html: bool,
                   written: list) -> None:
    """
    Figure 5: the same knee at two flexion angles — the pose, then the subdivisions.

    Three figures. The first draws both scans' cartilage voxels in world coordinates from
    the same camera, which shows how differently the two scans are posed. The next two
    draw the tibial and femoral subdivisions of each scan *in that bone's own frame*
    (the plate normals; the shaft axis and posterior condylar line), with the region
    shares in the panel titles. The regions are cut on the subchondral bone in that
    frame, so they land on the same anatomy in both scans and the shares barely move --
    which is what steps 2-3 of the plan bought and what the old tibial-plateau-window
    scheme could not do.

    :param scans: ``{scan_id: (tibia, femur)}`` for the two flexion angles, in the order
        they should be drawn.
    :param out_dir: Directory the files are written to.
    :param size: ``(width, height)`` in pixels.
    :param html: Also export each subdivision panel on its own as rotatable HTML, plus a
        page putting the two side by side (:func:`_panel_html`); the PNGs themselves are
        multi-panel and cannot be exported interactively.
    :param written: List the written paths are appended to.
    """
    ids = list(scans)
    tag = "_vs_".join(ids)

    # All comparison panels are drawn at one magnification (see _look_at's ``scale``), so
    # that what differs between two panels is the anatomy and not the framing. Each scan
    # is fitted in its own viewing frame first and the largest fit is used for all of
    # them: fitting one scan's bone in the other's frame would see it obliquely and
    # demand a frame that neither panel needs.
    aspect = size[1] / (size[0] / len(ids))

    def clouds(index):
        """Both scans' tAB point clouds for one bone (0 = tibia, 1 = femur)."""
        return [np.asarray(scans[scan][index].subdivision.mesh.points, dtype=float)
                for scan in ids]

    joint_clouds = [np.vstack(pair) for pair in zip(clouds(0), clouds(1))]
    joint_scale = max(_fit_scale(cloud, WORLD_RIGHT, WORLD_SUPERIOR, aspect)
                      for cloud in joint_clouds)
    tibia_scale = max(_fit_scale(cloud, *_plate_view(scans[scan][0].subdivision), aspect)
                      for scan, cloud in zip(ids, clouds(0)))
    femur_scale = max(_fit_scale(cloud, *_femur_view(scans[scan][1].subdivision,
                                                     "inferior"), aspect)
                      for scan, cloud in zip(ids, clouds(1)))

    # (a) the raw pose: same viewing direction and scale, world coordinates.
    plotter = _plotter(size, shape=(1, len(ids)))
    for column, scan in enumerate(ids):
        tibia, femur = scans[scan]
        plotter.subplot(0, column)
        plot_knee_segments(tibia, femur, plotter=plotter, show=False)
        _both_legend(plotter, tibia.subdivision, femur.subdivision)
        _add_world_axes(plotter)
        _title(plotter, f"{scan} — cartilage voxels in world (RAS) coordinates, seen "
                        f"from the patient's right\nsame viewing direction and scale in "
                        f"both panels: this is how differently the two scans are posed")
        _look_at(plotter, joint_clouds[column], WORLD_RIGHT, WORLD_SUPERIOR,
                 zoom=0.85, scale=joint_scale)
    _export(plotter, out_dir / f"flexion_pose_{tag}.png", None, written)

    def draw_tibia(plotter, scan, scale):
        """Draw one scan's tibial subdivision into the plotter's active panel."""
        tibia, _ = scans[scan]
        sub = tibia.subdivision
        shares = sub.region_shares
        tibia.plot_regions(plotter=plotter, show=False, show_bone=True, show_frames=False)
        add_tibia_construction(plotter, sub)
        _relegend(plotter, _legend_entries(sub, TIBIA_REGION_KEYS, TIBIA_COLORS,
                                           tibia_display_label))
        view_dir, up = _plate_view(sub)
        _add_region_labels(plotter, sub, TIBIA_REGION_KEYS, view_dir, 6.0,
                           display=tibia_display_label)
        _add_world_axes(plotter)
        left = ", ".join(f"{tibia_display_label(k)} {100 * shares[k]:.1f}"
                         for k in TIBIA_REGION_KEYS if k[1] == "l")
        right = ", ".join(f"{tibia_display_label(k)} {100 * shares[k]:.1f}"
                          for k in TIBIA_REGION_KEYS if k[1] == "r")
        _title(plotter, f"{scan} — tibial tAB in its own plate frame (superior, anterior up)\n"
                        f"shares of each plate's tAB (%): {left}\n{right}")
        _look_at(plotter, sub.mesh.points, view_dir, up, scale=scale)

    plotter = _plotter(size, shape=(1, len(ids)))
    for column, scan in enumerate(ids):
        plotter.subplot(0, column)
        draw_tibia(plotter, scan, tibia_scale)
    _export(plotter, out_dir / f"flexion_tibia_regions_{tag}.png", None, written)
    _panel_html(draw_tibia, ids, tibia_scale, size, html,
                out_dir, f"flexion_tibia_regions_{tag}", written)

    # (c) the femoral subdivision, each in its own femoral frame.
    def draw_femur(plotter, scan, scale):
        """Draw one scan's femoral subdivision into the plotter's active panel."""
        _, femur = scans[scan]
        sub = femur.subdivision
        shares = sub.region_shares
        femur.plot_regions(plotter=plotter, show=False, show_bone=True, show_frames=False)
        add_femur_construction(plotter, sub)
        _relegend(plotter, _legend_entries(sub, FEMUR_REGION_KEYS, FEMUR_COLORS))
        view_dir, up = _femur_view(sub, "inferior")
        _add_region_labels(plotter, sub, FEMUR_REGION_KEYS, view_dir, 7.0)
        _add_world_axes(plotter)
        thirds = ", ".join(f"{k} {100 * shares[k]:.1f}" for k in sorted(shares))
        _title(plotter, f"{scan} — femoral tAB in the femoral frame (from inferior, "
                        f"anterior up)\nthirds' shares of their condyle's cF (%): {thirds}")
        _look_at(plotter, sub.mesh.points, view_dir, up, scale=scale)

    plotter = _plotter(size, shape=(1, len(ids)))
    for column, scan in enumerate(ids):
        plotter.subplot(0, column)
        draw_femur(plotter, scan, femur_scale)
    _export(plotter, out_dir / f"flexion_femur_regions_{tag}.png", None, written)
    _panel_html(draw_femur, ids, femur_scale, size, html,
                out_dir, f"flexion_femur_regions_{tag}", written)


def figure_thinnest(tibia: Tibia, femur: Femur, tibia_thickness, femur_thickness,
                    scan: str, out_dir: Path, size: tuple, html: bool,
                    written: list) -> None:
    """
    Figure 6: the thinnest cartilage point of the whole knee and of each compartment.

    ``plot_thinnest_point`` draws one arrow per entry of a ``{label: result}`` dict over
    the thickness heatmap: ``both`` (black) from :func:`find_thinnest_area` over the
    whole tibiofemoral overlap, and ``left`` (red) / ``right`` (blue) from
    :func:`find_thinnest_area_by_compartment`. **The compartment labels are patient
    sides, not medial / lateral.** A compartment whose search fails is reported and
    skipped rather than aborting the figure.

    :param tibia: The subdivided ``Tibia``.
    :param femur: The subdivided ``Femur`` of the same knee.
    :param tibia_thickness: The tibial ``ThicknessMaps`` (reused from
        :func:`figure_thickness`).
    :param femur_thickness: The femoral ``ThicknessMaps``.
    :param scan: Scan id, used in the title and the file name.
    :param out_dir: Directory the files are written to.
    :param size: ``(width, height)`` in pixels.
    :param html: Also export interactive HTML.
    :param written: List the written paths are appended to.
    """
    results = {}
    try:
        results["both"] = find_thinnest_area(tibia_thickness, femur_thickness, tibia.image)
    except (RuntimeError, AssertionError, ValueError) as exc:
        print(f"  [warning] whole-knee thinnest point failed: {exc}", file=sys.stderr)
    try:
        results.update(find_thinnest_area_by_compartment(
            tibia_thickness, femur_thickness, tibia.image))
    except (RuntimeError, AssertionError, ValueError) as exc:
        print(f"  [warning] per-compartment thinnest point failed: {exc}", file=sys.stderr)
    if not results:
        print("  [warning] no thinnest point to draw; skipping the figure.", file=sys.stderr)
        return
    for label, result in results.items():
        print(f"  thinnest ({label}): combined {result['combined_thickness']:.3f} mm "
              f"(smoothed {result['center_thickness']:.3f}), tibial "
              f"{result['tibial_thickness']:.3f} + femoral {result['femoral_thickness']:.3f}, "
              f"kernel {result['kernel_size']}")

    # The whole-knee minimum is by construction one of the two compartments' minima, so
    # the black arrow sits exactly under that compartment's and is invisible. Say which.
    coincides = [label for label in ("left", "right")
                 if label in results and "both" in results
                 and np.array_equal(results[label]["point"], results["both"]["point"])]
    note = (f"\nthe whole-knee point is the patient-{coincides[0].upper()} one, so the "
            f"black arrow is hidden under that compartment's" if coincides else "")

    points = np.vstack([np.asarray(tibia.subdivision.mesh.points, dtype=float),
                        np.asarray(femur.subdivision.mesh.points, dtype=float)])
    plotter = _plotter(size)
    plot_thinnest_point(tibia, femur, tibia_thickness, femur_thickness, results,
                        plotter=plotter, show=False, base="thickness")
    _add_world_axes(plotter)
    _title(plotter, (
        f"Thinnest tibiofemoral cartilage — {scan}\n"
        f"black = whole knee, red = patient-LEFT compartment, blue = patient-RIGHT "
        f"(patient sides, NOT medial/lateral)\n"
        + "  ".join(f"{label}: {result['combined_thickness']:.2f} mm"
                    for label, result in results.items()) + note))
    # Zoomed out: the arrows are scaled to the scene by the library and reach well beyond
    # the cartilage, so a frame fitted to the cartilage alone would cut their tails off.
    _look_at(plotter, points, WORLD_ANTERIOR + 0.45 * WORLD_RIGHT, WORLD_SUPERIOR,
             zoom=0.7)
    _export(plotter, out_dir / f"knee_thinnest_point_{scan}.png",
            out_dir / f"knee_thinnest_point_{scan}.html" if html else None, written)


# ----------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------

ALL_FIGURES = ("tibia", "femur", "both", "thickness", "flexion", "thinnest")


def _write_gallery(paths: list, out_dir: Path, scan: str, written: list) -> None:
    """
    Write ``index.html``: one page listing every figure of a run, grouped by figure.

    The run leaves up to five files per figure (PNG, GIF, MP4, one or more interactive
    scenes), which is a directory listing nobody wants to read. The gallery embeds the
    still, offers the turntable inline where one was filmed, and links the rotatable
    scenes, so a reader opens one file.

    :param paths: Everything written so far, in order (``index.html`` itself excluded).
    :param out_dir: The output directory; the page is written there and links are
        relative to it.
    :param scan: The reference scan id, for the page title.
    :param written: A list the written path is appended to, for the final report.
    """
    groups = {}
    for path in paths:
        if path.suffix.lower() not in (".png", ".gif", ".mp4", ".html"):
            continue
        groups.setdefault(path.stem.replace(".panel", ""), {})[path.suffix.lower()] = path

    # A comparison figure's per-panel scenes (``<stem>_<scan>.html``) are reached from its
    # own side-by-side page, so they would only appear here as near-duplicate, still-less
    # sections of the figure they belong to.
    stems = set(groups)
    groups = {stem: files for stem, files in groups.items()
              if set(files) != {".html"}
              or not any(stem != other and stem.startswith(other + "_") for other in stems)}

    sections = []
    for stem, files in groups.items():
        parts = [f"<h2>{stem}</h2>"]
        if ".png" in files:
            parts.append(f'<a href="{files[".png"].name}">'
                         f'<img src="{files[".png"].name}" loading="lazy"></a>')
        if ".mp4" in files:
            parts.append(f'<video src="{files[".mp4"].name}" controls loop muted '
                         f'playsinline></video>')
        elif ".gif" in files:
            parts.append(f'<img src="{files[".gif"].name}" loading="lazy">')
        links = [f'<a href="{files[suffix].name}">{label}</a>'
                 for suffix, label in ((".html", "rotate"), (".gif", "GIF"),
                                       (".mp4", "MP4")) if suffix in files]
        if links:
            parts.append(f'<p>{" · ".join(links)}</p>')
        sections.append("<section>" + "".join(parts) + "</section>")

    path = out_dir / "index.html"
    path.write_text(
        "<!doctype html><meta charset='utf-8'>"
        f"<title>Knee cartilage zones — {scan}</title>"
        "<style>body{font:14px/1.5 system-ui,sans-serif;margin:1.5rem;background:#e9e9e9;"
        "color:#222}section{margin-bottom:2.5rem}h2{font-size:1rem;font-family:monospace}"
        "img,video{max-width:100%;border:1px solid #bbb;background:#fff}"
        "a{color:#00429d}</style>"
        f"<h1>Knee cartilage subregions — {scan}</h1>"
        "<p>Wirth &amp; Eckstein 2008 subdivision on the subchondral bone area. Region "
        "keys are <b>patient sides</b>, not medial/lateral.</p>"
        + "".join(sections),
        encoding="utf-8")
    print(f"wrote {path}")
    written.append(path)


def render(data_dir: Path, scan: str, flexion_scan: str, figures: tuple, out_dir: Path,
           size: tuple, html: bool, laterality: str = None,
           allow_flagged: bool = False, animate: dict = None) -> list:
    """
    Load the scan(s), subdivide them and write the requested figures.

    The reference scan is always loaded; the flexion scan only when the ``flexion``
    figure is requested. The thickness maps are measured once and shared by the
    ``thickness`` and ``thinnest`` figures (they are the slow part, ~15 s per knee).

    :param data_dir: Directory holding the ``dfg_knees`` segmentations.
    :param scan: The reference scan id.
    :param flexion_scan: The second scan of the same knee, at a different flexion angle.
    :param figures: Which figures to draw (subset of :data:`ALL_FIGURES`).
    :param out_dir: Directory the figures are written to.
    :param size: ``(width, height)`` in pixels.
    :param html: Also export interactive HTML where a single-renderer scene allows it.
    :param laterality: ``'left'`` / ``'right'`` knee, or ``None`` (see
        :func:`_laterality_note`): an assumption, not data.
    :param allow_flagged: Permit a QC-flagged scan.
    :param animate: ``{'frames': int, 'fps': int}`` to film a turntable of every figure
        alongside its PNG, or ``None`` for stills only.
    :return: The paths written, in order.
    """
    global _ANIMATION
    _ANIMATION = animate
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    seg = load_scan(data_dir, scan, allow_flagged)
    tibia, femur = build_bones(seg, laterality)
    report_shares(scan, tibia, femur)

    thickness = None
    if "tibia" in figures:
        figure_tibia(tibia, scan, out_dir, size, html, written)
    if "femur" in figures:
        figure_femur(femur, scan, out_dir, size, html, written)
    if "both" in figures:
        figure_both(tibia, femur, scan, out_dir, size, html, written)
    if "thickness" in figures or "thinnest" in figures:
        thickness = figure_thickness(tibia, femur, scan, out_dir, size, html, written)
    if "thinnest" in figures:
        figure_thinnest(tibia, femur, thickness[0], thickness[1], scan, out_dir, size,
                        html, written)
    if "flexion" in figures:
        flexion_seg = load_scan(data_dir, flexion_scan, allow_flagged)
        flexion_bones = build_bones(flexion_seg, laterality)
        report_shares(flexion_scan, *flexion_bones)
        figure_flexion({scan: (tibia, femur), flexion_scan: flexion_bones},
                       out_dir, size, html, written)

    if html:
        _write_gallery(written, out_dir, scan, written)

    print(f"\n{len(written)} file(s) in {out_dir}:")
    for path in written:
        print(f"  {path}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                    help=f"dfg_knees segmentation directory (default: {DEFAULT_DATA_DIR}; "
                         f"env MORPH_DFG_KNEES_DIR)")
    ap.add_argument("--scan", default=DEFAULT_SCAN,
                    help=f"reference scan id without suffix (default: {DEFAULT_SCAN})")
    ap.add_argument("--flexion-scan", default=DEFAULT_FLEXION_SCAN,
                    help=f"the same knee at another flexion angle, for the 'flexion' "
                         f"figure (default: {DEFAULT_FLEXION_SCAN})")
    ap.add_argument("--figures", nargs="+", choices=ALL_FIGURES + ("all",), default=["all"],
                    help=f"which figures to draw (default: all = {' '.join(ALL_FIGURES)})")
    ap.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT,
                    help=f"output directory (default: {DEFAULT_OUT}, gitignored)")
    ap.add_argument("--size", type=int, nargs=2, default=[1600, 1200],
                    metavar=("WIDTH", "HEIGHT"),
                    help="PNG size in pixels (default: 1600 1200; multi-panel figures use "
                         "it for the whole figure)")
    ap.add_argument("--no-html", action="store_true",
                    help="skip the interactive HTML exports (they are ~5-20 MB each)")
    ap.add_argument("--laterality", choices=["left", "right"], default=None,
                    help="knee laterality, to add the paper's medial/lateral names. It is "
                         "NOT recorded for dfg_knees (plan open issue 3), so this is an "
                         "assumption and is marked as one in the figure titles.")
    ap.add_argument("--allow-flagged", action="store_true",
                    help=f"permit a QC-flagged scan ({', '.join(FLAGGED_SCANS)})")
    ap.add_argument("--animate", action="store_true",
                    help="also film a turntable of every figure drawn: each panel spins "
                         "about its own anatomical up vector, written as a GIF and (where "
                         "ffmpeg is on PATH) an MP4 beside the PNG")
    ap.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                    help=f"frames in one revolution (default: {DEFAULT_FRAMES})")
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS,
                    help=f"turntable playback rate (default: {DEFAULT_FPS})")
    args = ap.parse_args()

    figures = tuple(ALL_FIGURES) if "all" in args.figures else tuple(
        f for f in ALL_FIGURES if f in args.figures)
    animate = {"frames": args.frames, "fps": args.fps} if args.animate else None
    render(args.data_dir, args.scan, args.flexion_scan, figures, args.out,
           tuple(args.size), not args.no_html, laterality=args.laterality,
           allow_flagged=args.allow_flagged, animate=animate)


if __name__ == "__main__":
    main()
