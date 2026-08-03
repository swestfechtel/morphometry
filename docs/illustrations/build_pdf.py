#!/usr/bin/env python3
"""Convert the parameter illustration SVGs to vector PDFs for the LaTeX build.

The SVGs style themselves with a CSS ``<style>`` block, which most SVG->PDF
converters (rsvg, MuPDF) ignore — only a real browser engine renders them
correctly. So we drive headless Chrome's ``--print-to-pdf`` with an HTML wrapper
whose ``@page`` size matches each SVG, yielding a cropped, vector PDF (one page,
mediabox = the SVG box in points). Output goes to ``illustrations/pdf/*.pdf``,
which ``reader_measurement_reference_table.tex`` includes.

    python docs/illustrations/build_pdf.py

Run after ``generate_illustrations.py``. Requires google-chrome / chromium.
"""
import glob
import os
import re
import shutil
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(HERE, "pdf")


def find_chrome():
    for name in ("google-chrome", "chromium", "chromium-browser", "google-chrome-stable"):
        path = shutil.which(name)
        if path:
            return path
    raise SystemExit("No Chrome/Chromium found — cannot render SVG to vector PDF.")


def convert(chrome, svg_path, out_path, workdir):
    svg = open(svg_path).read()
    w = re.search(r'width="(\d+)"', svg).group(1)
    h = re.search(r'height="(\d+)"', svg).group(1)
    html = (f'<!doctype html><html><head><meta charset="utf-8"><style>\n'
            f'@page {{ size: {w}px {h}px; margin: 0; }}\n'
            f'html,body {{ margin:0; padding:0; }} svg {{ display:block; }}\n'
            f'</style></head><body>{svg}</body></html>')
    wrapper = os.path.join(workdir, "wrap.html")
    with open(wrapper, "w") as f:
        f.write(html)
    subprocess.run(
        [chrome, "--headless", "--disable-gpu", "--no-sandbox",
         "--no-pdf-header-footer", f"--print-to-pdf={out_path}", wrapper],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def main():
    chrome = find_chrome()
    os.makedirs(PDF_DIR, exist_ok=True)
    with tempfile.TemporaryDirectory() as workdir:
        for svg_path in sorted(glob.glob(os.path.join(HERE, "*.svg"))):
            name = os.path.splitext(os.path.basename(svg_path))[0]
            out_path = os.path.join(PDF_DIR, name + ".pdf")
            convert(chrome, svg_path, out_path, workdir)
            print("wrote", os.path.relpath(out_path, HERE))


if __name__ == "__main__":
    main()
