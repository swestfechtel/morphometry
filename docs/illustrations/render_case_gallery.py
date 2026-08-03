#!/usr/bin/env python3
"""Batch-render the subchondral ray tracing across several NaKo cases + a contact sheet.

Renders both hips of the first ``--n-cases`` sample segmentations at a fixed crop
factor and tiles the PNGs into one contact sheet, so a crop choice
(``acetabulum_radius_factor``) can be eyeballed across cases rather than on a single
hip. Individual PNGs and the sheet go under ``docs/illustrations/cases/`` (gitignored).

    MORPH_NAKO_SAMPLE_DIR=/path/to/segmentations \
        python docs/illustrations/render_case_gallery.py --radius-factor 1.5 --n-cases 10

Needs the sample data and a PyVista off-screen backend (same as plot_subchondral_3d.py).
"""
import argparse
from pathlib import Path

from PIL import Image as PILImage

from plot_subchondral_3d import HERE, _load_sample, _sample_files, _split_side, render_case

CASES_DIR = HERE / "cases"


def _autocrop(img: PILImage.Image, bg=(255, 255, 255), margin: int = 8) -> PILImage.Image:
    """Trim a uniform (white) border so the joint fills the panel, leaving a small margin."""
    diff = PILImage.new("RGB", img.size, bg)
    from PIL import ImageChops
    bbox = ImageChops.difference(img.convert("RGB"), diff).getbbox()
    if bbox is None:
        return img
    l, t, r, b = bbox
    l, t = max(0, l - margin), max(0, t - margin)
    r, b = min(img.width, r + margin), min(img.height, b + margin)
    return img.crop((l, t, r, b))


def _contact_sheet(panels: list[Path], cols: int, panel_w: int = 520) -> PILImage.Image:
    """Tile PNG panels into a grid, each autocropped and scaled to a uniform width."""
    imgs = []
    for p in panels:
        im = _autocrop(PILImage.open(p))
        h = int(im.height * panel_w / im.width)
        imgs.append(im.resize((panel_w, h)))
    panel_h = max(im.height for im in imgs)
    rows = (len(imgs) + cols - 1) // cols
    sheet = PILImage.new("RGB", (cols * panel_w, rows * panel_h), (255, 255, 255))
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        sheet.paste(im, (c * panel_w, r * panel_h))
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--radius-factor", type=float, default=1.5,
                    help="peri-acetabular crop radius in head radii (<=0 disables)")
    ap.add_argument("--n-cases", type=int, default=10, help="number of cases (both hips each)")
    ap.add_argument("--cols", type=int, default=4, help="columns in the contact sheet")
    ap.add_argument("--html", action="store_true",
                    help="also export a rotatable .html next to each case PNG (~7-8 MB each)")
    args = ap.parse_args()
    radius_factor = args.radius_factor if args.radius_factor > 0 else None
    tag = f"{radius_factor:g}r".replace(".", "p") if radius_factor is not None else "full"

    CASES_DIR.mkdir(exist_ok=True)
    n = min(args.n_cases, len(_sample_files()))
    panels, stats = [], []
    for idx in range(n):
        seg, case_id = _load_sample(idx)
        for side in ("left", "right"):
            out = CASES_DIR / f"case{idx:02d}_{case_id}_{side}_{tag}.png"
            stats.append(render_case(_split_side(seg, side), case_id, side, radius_factor,
                                     str(out), html=args.html))
            panels.append(out)

    sheet = _contact_sheet(panels, cols=args.cols)
    sheet_path = CASES_DIR / f"contact_sheet_{tag}.png"
    sheet.save(sheet_path)
    print("wrote", sheet_path)

    # a compact table for the terminal
    print(f"\ncrop {tag}  ({n} cases, both hips)")
    print(f"{'case':>8} {'side':>5} {'mean':>6} {'std':>5} {'min':>5} {'max':>6} {'rays':>5}")
    for s in stats:
        print(f"{s['case']:>8} {s['side']:>5} {s['mean']:6.2f} {s['std']:5.2f} "
              f"{s['min']:5.2f} {s['max']:6.2f} {s['rays']:5d}")


if __name__ == "__main__":
    main()
