#!/usr/bin/env python3
"""Generate schematic SVG illustrations for each morphometric parameter.

One SVG per parameter in ``docs/reader_measurement_reference_table.md`` /
``.tex``. These are didactic diagrams — simplified bone shapes plus the reference
lines, landmarks and the measured angle / distance — sharing one visual system so
they read as a set. Regenerate with:

    python docs/illustrations/generate_illustrations.py

Geometry (arc sweeps, perpendicular feet) is computed rather than hand-placed so
the drawn angle actually matches the anatomy. Colours: proximal/first reference
= blue, distal/second reference = red, construction = dashed grey, measured
angle = emerald, measured distance = purple. Keep captions short enough to fit
the canvas width, and never use a bare ``&`` (XML entity) in text.
"""
import math
import os

OUT = os.path.dirname(os.path.abspath(__file__))

STYLE = """
  <style>
    .bg{fill:#ffffff}
    .bone{fill:#ECE3D0;stroke:#A9946A;stroke-width:2;stroke-linejoin:round}
    .bone2{fill:#F5EFE1;stroke:#C3B48E;stroke-width:1.6;stroke-linejoin:round}
    .A{stroke:#2563EB;stroke-width:3;fill:none;stroke-linecap:round}
    .B{stroke:#DC2626;stroke-width:3;fill:none;stroke-linecap:round}
    .C{stroke:#94A3B8;stroke-width:1.6;stroke-dasharray:6 4;fill:none}
    .arc{stroke:#0E9F6E;stroke-width:2.6;fill:none}
    .dist{stroke:#7C3AED;stroke-width:2.6;fill:none}
    .ptl{fill:#334155;stroke:#fff;stroke-width:1.5}
    .t{font:700 15px system-ui,-apple-system,Segoe UI,Roboto,sans-serif;fill:#0F172A}
    .s{font:400 11px system-ui,-apple-system,Segoe UI,Roboto,sans-serif;fill:#64748B}
    .lA{font:600 12px system-ui,sans-serif;fill:#2563EB}
    .lB{font:600 12px system-ui,sans-serif;fill:#DC2626}
    .lC{font:600 12px system-ui,sans-serif;fill:#7C3AED}
    .v{font:800 15px system-ui,sans-serif;fill:#0E9F6E}
    .l{font:500 12px system-ui,sans-serif;fill:#334155}
  </style>
"""

DEFS = """
  <defs>
    <marker id="ah" markerWidth="10" markerHeight="10" refX="7.5" refY="5" orient="auto">
      <path d="M0 0 L10 5 L0 10 z" fill="#7C3AED"/></marker>
    <marker id="ahs" markerWidth="10" markerHeight="10" refX="2.5" refY="5" orient="auto">
      <path d="M10 0 L0 5 L10 10 z" fill="#7C3AED"/></marker>
  </defs>
"""


def hdr(w, h, title, sub):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}">\n'
            f'  <rect class="bg" x="0" y="0" width="{w}" height="{h}"/>{STYLE}{DEFS}\n'
            f'  <text class="t" x="16" y="28">{title}</text>\n'
            f'  <text class="s" x="16" y="46">{sub}</text>\n')


def foot(w, h, caption):
    return f'  <text class="s" x="16" y="{h - 14}">{caption}</text>\n</svg>\n'


def line(x1, y1, x2, y2, cls):
    return f'  <line class="{cls}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>\n'


def circle(cx, cy, r, cls="bone"):
    return f'  <circle class="{cls}" cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}"/>\n'


def ellipse(cx, cy, rx, ry, cls="bone", rot=0.0):
    tr = f' transform="rotate({rot:.1f} {cx:.1f} {cy:.1f})"' if rot else ""
    return f'  <ellipse class="{cls}" cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}"{tr}/>\n'


def poly(pts, cls="bone"):
    p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return f'  <polygon class="{cls}" points="{p}"/>\n'


def ppath(d, cls="bone"):
    return f'  <path class="{cls}" d="{d}"/>\n'


def dot(x, y, r=4.5):
    return f'  <circle class="ptl" cx="{x:.1f}" cy="{y:.1f}" r="{r}"/>\n'


def txt(x, y, s, cls="l", anchor="start"):
    return f'  <text class="{cls}" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}">{s}</text>\n'


def _ang(cx, cy, px, py):
    return math.atan2(py - cy, px - cx)


def arc(cx, cy, r, pfrom, pto):
    """Emerald angle arc from the ray toward pfrom to the ray toward pto (short way)."""
    a1 = _ang(cx, cy, *pfrom)
    a2 = _ang(cx, cy, *pto)
    d = a2 - a1
    while d <= -math.pi:
        d += 2 * math.pi
    while d > math.pi:
        d -= 2 * math.pi
    sweep = 1 if d > 0 else 0
    x1, y1 = cx + r * math.cos(a1), cy + r * math.sin(a1)
    x2, y2 = cx + r * math.cos(a2), cy + r * math.sin(a2)
    return f'  <path class="arc" d="M{x1:.1f} {y1:.1f} A{r} {r} 0 0 {sweep} {x2:.1f} {y2:.1f}"/>\n'


def darrow(x1, y1, x2, y2):
    return (f'  <line class="dist" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'marker-start="url(#ahs)" marker-end="url(#ah)"/>\n')


def foot_perp(p, a, b):
    """Foot of perpendicular from point p onto the infinite line through a,b."""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)
    return (ax + t * dx, ay + t * dy)


def extend(a, b, k):
    """Point b extended by k px further along the a->b direction."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy)
    return (b[0] + dx / n * k, b[1] + dy / n * k)


def neck_poly(a, b, hw, shift=0.0):
    """A bone-coloured quad of half-width hw around the segment a->b."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    px, py = -uy * hw, ux * hw
    return poly([(a[0] + px, a[1] + py), (b[0] + px + shift, b[1] + py),
                 (b[0] - px + shift, b[1] - py), (a[0] - px, a[1] - py)])


# ---------------------------------------------------------------------------
# 1. Femoral torsion
# ---------------------------------------------------------------------------
def femoral_torsion():
    W, H = 380, 404
    s = hdr(W, H, "Femoral torsion (anteversion)",
            "Transverse plane — neck axis vs posterior condylar line")
    Hc = (120, 152)
    N = (234, 186)
    s += neck_poly((Hc[0] + 16, Hc[1] + 6), N, 12)
    s += circle(*Hc, 30)
    s += ellipse(150, 322, 30, 37)
    s += ellipse(216, 322, 30, 37)
    s += line(190, 214, 190, 276, "C")                 # superimposition hint
    s += line(112, 352, 254, 352, "B")                 # posterior condylar line (distal)
    s += dot(112, 352, 4)
    s += dot(254, 352, 4)
    axis_end = extend(Hc, N, 26)
    s += line(Hc[0], Hc[1], axis_end[0], axis_end[1], "A")   # neck axis (proximal)
    s += line(Hc[0], Hc[1], 266, Hc[1], "C")                 # horizontal reference
    s += arc(Hc[0], Hc[1], 66, (266, Hc[1]), axis_end)
    s += dot(*Hc)
    s += dot(*N)
    s += txt(196, 176, "≈15°", "v")
    s += txt(axis_end[0] + 6, axis_end[1] + 4, "neck axis", "lA")
    s += txt(254, 372, "condylar line", "lB", "end")
    s += foot(W, H, "Neck axis vs posterior condylar line, transverse plane.")
    return "femoral_torsion.svg", s


# ---------------------------------------------------------------------------
# 2. Tibial torsion
# ---------------------------------------------------------------------------
def tibial_torsion():
    W, H = 380, 404
    s = hdr(W, H, "Tibial torsion",
            "Transverse plane — proximal condylar vs malleolar line")
    s += ellipse(178, 150, 74, 30)                     # proximal tibial plateau
    s += line(112, 168, 244, 168, "A")                 # proximal posterior condylar line
    s += dot(112, 168, 4)
    s += dot(244, 168, 4)
    s += line(178, 186, 178, 268, "C")                 # superimposition
    T = (150, 330)
    F = (206, 296)
    s += ellipse(T[0], T[1], 37, 31)                   # tibia
    s += circle(F[0], F[1], 16, "bone")                # fibula
    end = extend(T, F, 18)
    s += line(T[0], T[1], end[0], end[1], "B")         # distal trans-malleolar line
    s += line(T[0], T[1], 278, T[1], "C")              # horizontal reference
    s += arc(T[0], T[1], 72, (278, T[1]), end)
    s += dot(*T)
    s += dot(*F)
    s += txt(212, 316, "≈30°", "v")
    s += txt(112, 158, "proximal condylar line", "lA")
    s += txt(end[0] + 6, end[1] + 2, "malleolar line", "lB")
    s += foot(W, H, "Proximal condylar vs distal trans-malleolar line.")
    return "tibial_torsion.svg", s


# ---------------------------------------------------------------------------
# 3. Knee rotation angle
# ---------------------------------------------------------------------------
def knee_rotation():
    W, H = 380, 372
    s = hdr(W, H, "Knee rotation angle",
            "Transverse plane — femoral vs tibial condylar line")
    s += ellipse(150, 150, 28, 33)                     # femoral condyles
    s += ellipse(214, 150, 28, 33)
    s += line(116, 180, 250, 180, "A")                 # femoral posterior condylar line
    s += dot(116, 180, 4)
    s += dot(250, 180, 4)
    s += ellipse(182, 262, 74, 30)                     # tibial plateau
    s += line(116, 240, 250, 250, "B")                 # tibial posterior condylar line
    s += dot(116, 240, 4)
    s += dot(250, 250, 4)
    V = (150, 210)                                      # measured-angle motif
    r1 = (285, 210)
    r2 = (287, 197)
    s += line(V[0], V[1], r1[0], r1[1], "C")
    s += line(V[0], V[1], r2[0], r2[1], "C")
    s += arc(V[0], V[1], 66, r1, r2)
    s += dot(*V)
    s += txt(224, 203, "small (few°)", "v")
    s += txt(120, 172, "femoral line", "lA")
    s += txt(120, 232, "tibial line", "lB")
    s += foot(W, H, "Femoro-tibial rotational mismatch; near-parallel in extension.")
    return "knee_rotation.svg", s


# ---------------------------------------------------------------------------
# 4. CCD (neck–shaft) angle
# ---------------------------------------------------------------------------
def ccd_angle():
    W, H = 360, 432
    s = hdr(W, H, "CCD angle (neck–shaft)",
            "Coronal plane — femoral neck axis vs shaft axis")
    Hc = (98, 164)
    V = (182, 214)                                     # neck/shaft intersection
    shaft_bot = (210, 418)
    s += poly([(156, 196), (206, 188), (232, 418), (188, 418)])       # shaft bone
    s += ppath("M196 176 Q236 168 226 210 Q206 206 196 196 Z")       # greater trochanter
    s += neck_poly(Hc, V, 18, shift=8)                                # neck bone
    s += circle(*Hc, 30)                                              # femoral head
    neck_end = extend(V, Hc, -6)
    s += line(V[0], V[1], neck_end[0], neck_end[1], "A")             # neck axis
    s += line(180, 176, shaft_bot[0], shaft_bot[1], "B")            # shaft axis
    s += arc(V[0], V[1], 52, Hc, shaft_bot)
    s += dot(*Hc)
    s += dot(*V, 4)
    s += txt(104, 258, "≈127°", "v")
    s += txt(66, 138, "neck axis", "lA")
    s += txt(232, 320, "shaft axis", "lB", "end")
    s += foot(W, H, "Neck–shaft angle, reported as the obtuse value.")
    return "ccd_angle.svg", s


# ---------------------------------------------------------------------------
# 5. Bone length
# ---------------------------------------------------------------------------
def bone_length():
    W, H = 300, 520
    s = hdr(W, H, "Bone length (femur)",
            "3D centroid-to-centroid distance")
    s += circle(112, 108, 28)                          # head
    s += ppath("M150 96 Q182 96 176 130 Q160 138 146 126 Z")     # trochanter
    s += poly([(126, 132), (168, 140), (150, 452), (120, 452)])  # shaft
    s += ellipse(126, 470, 26, 24)                     # medial condyle
    s += ellipse(168, 470, 26, 24)                     # lateral condyle
    P = (140, 118)
    D = (147, 468)
    s += line(P[0], P[1], D[0], D[1], "C")
    s += dot(*P)
    s += dot(*D)
    s += txt(150, 112, "proximal centroid", "l")
    s += txt(155, 486, "distal centroid", "l")
    xb = 236
    s += line(P[0], P[1], xb, P[1], "C")
    s += line(D[0], D[1], xb, D[1], "C")
    s += darrow(xb, P[1], xb, D[1])
    s += txt(xb + 8, (P[1] + D[1]) / 2, "≈500 mm", "lC")
    s += foot(W, H, "Euclidean length between end-slice centroids.")
    return "bone_length.svg", s


# ---------------------------------------------------------------------------
# 6. Acetabular version
# ---------------------------------------------------------------------------
def acetabular_version():
    W, H = 380, 372
    s = hdr(W, H, "Acetabular version (anteversion)",
            "Transverse slice — anterior is up")
    Lh = (108, 196)
    Rh = (272, 196)
    s += circle(*Lh, 26)
    s += circle(*Rh, 26)
    # acetabular rim crescent lateral to the right head
    s += ppath("M272 150 Q334 152 330 196 Q332 240 300 234 "
               "L306 222 Q318 196 314 174 Q308 158 278 160 Z")
    s += line(Lh[0], Lh[1], Rh[0], Rh[1], "A")         # reference line G
    p1 = (300, 234)                                    # posterior rim
    p2 = (272, 150)                                    # anterior rim
    s += line(p1[0], p1[1], p2[0], p2[1], "B")         # rim line
    perp_top = (p1[0], p1[1] - 100)                    # perpendicular to G through p1
    s += line(p1[0], p1[1], perp_top[0], perp_top[1], "C")
    s += arc(p1[0], p1[1], 66, perp_top, p2)
    s += dot(*p1)
    s += dot(*p2)
    s += dot(*Lh, 4)
    s += dot(*Rh, 4)
    s += txt(236, 164, "≈18°", "v")
    s += txt(150, 186, "line G", "lA")
    s += txt(310, 138, "rim line", "lB")
    s += txt(150, 330, "anterior ↑   posterior ↓", "s")
    s += foot(W, H, "Rim line vs the perpendicular to the inter-head line.")
    return "acetabular_version.svg", s


# ---------------------------------------------------------------------------
# 7. Center-edge (CE) angle
# ---------------------------------------------------------------------------
def ce_angle():
    W, H = 360, 424
    s = hdr(W, H, "Center-edge (CE) angle of Wiberg",
            "Coronal plane — lateral acetabular coverage")
    C = (156, 258)
    s += circle(*C, 46)                                # femoral head
    s += poly([(128, 300), (184, 300), (196, 410), (140, 410)])   # neck/shaft
    E = (192, 196)                                     # lateral acetabular edge
    s += ppath("M84 214 Q118 172 192 196 L196 214 Q126 196 96 232 Z")   # acetabular roof
    s += line(78, C[1], 236, C[1], "A")                # inter-head line G
    vtop = (C[0], C[1] - 110)                          # vertical perpendicular to G
    s += line(C[0], C[1], vtop[0], vtop[1], "C")
    s += line(C[0], C[1], E[0], E[1], "B")             # head-centre → lateral edge
    s += arc(C[0], C[1], 78, vtop, E)
    s += dot(*C)
    s += dot(*E)
    s += txt(120, 168, "≈33°", "v")
    s += txt(198, 192, "lateral edge", "lB")
    s += txt(80, C[1] + 18, "line G", "lA")
    s += foot(W, H, "Vertical reference vs head-centre→lateral-edge line.")
    return "ce_angle.svg", s


# ---------------------------------------------------------------------------
# 8. Femoral offset
# ---------------------------------------------------------------------------
def femoral_offset():
    W, H = 360, 432
    s = hdr(W, H, "Femoral offset",
            "Coronal plane — head centre to shaft axis")
    Hc = (104, 158)
    P1 = (176, 176)
    P2 = (198, 416)                                    # shaft axis endpoints
    s += poly([(156, 196), (200, 188), (222, 416), (178, 416)])   # shaft bone
    s += ppath("M192 176 Q232 168 222 210 Q202 206 192 196 Z")    # trochanter
    s += neck_poly(Hc, (168, 190), 18)                            # neck bone
    s += circle(*Hc, 30)
    F = foot_perp(Hc, P1, P2)
    a_top = extend(P2, P1, 48)                         # extend axis up to include the foot
    s += line(a_top[0], a_top[1], P2[0], P2[1], "A")   # shaft axis
    s += darrow(Hc[0], Hc[1], F[0], F[1])              # perpendicular offset
    s += dot(*Hc)
    s += dot(*F, 4)
    s += txt(36, 150, "head centre", "l")
    s += txt(210, 300, "shaft axis", "lA", "end")
    s += txt((Hc[0] + F[0]) / 2, 136, "≈43 mm", "lC", "middle")
    s += foot(W, H, "Perpendicular distance from head centre to shaft axis.")
    return "femoral_offset.svg", s


# ---------------------------------------------------------------------------
# 9. HKA angle
# ---------------------------------------------------------------------------
def hka_angle():
    W, H = 300, 520
    s = hdr(W, H, "Hip–knee–ankle (HKA) angle",
            "Coronal mechanical axis — varus / valgus")
    Hh = (118, 92)
    K = (150, 300)
    Ank = (140, 486)
    s += poly([(104, 100), (140, 108), (166, 296), (138, 300)], "bone2")   # femur
    s += poly([(140, 306), (166, 302), (152, 486), (128, 486)], "bone2")   # tibia
    s += circle(114, 96, 20, "bone2")                  # femoral head
    Cc = extend(Hh, K, 86)                             # straight continuation past knee
    s += line(K[0], K[1], Cc[0], Cc[1], "C")
    s += line(Hh[0], Hh[1], K[0], K[1], "A")           # hip → knee
    s += line(K[0], K[1], Ank[0], Ank[1], "B")         # knee → ankle
    s += arc(K[0], K[1], 58, Cc, Ank)
    s += dot(*Hh)
    s += dot(*K)
    s += dot(*Ank)
    s += txt(Hh[0] + 8, Hh[1] - 6, "hip", "lA")
    s += txt(K[0] + 12, K[1] - 6, "knee", "l")
    s += txt(Ank[0] + 8, Ank[1] + 6, "ankle", "lB")
    s += txt(K[0] + 22, K[1] + 44, "≈0° (varus)", "v")
    s += foot(W, H, "Bend at the knee; 0° = perfectly straight.")
    return "hka_angle.svg", s


# ---------------------------------------------------------------------------
# 10. JLCA
# ---------------------------------------------------------------------------
def jlca():
    W, H = 380, 360
    s = hdr(W, H, "Joint-line convergence angle (JLCA)",
            "Coronal knee — femoral vs tibial joint line")
    s += ellipse(150, 132, 30, 34)                     # femoral condyles
    s += ellipse(214, 132, 30, 34)
    s += ellipse(182, 250, 78, 28)                     # tibial plateau
    A1, A2 = (104, 172), (250, 190)                    # distal femoral joint line
    B1, B2 = (104, 214), (250, 196)                    # proximal tibial joint line
    s += line(*A1, *A2, "A")
    s += line(*B1, *B2, "B")

    def inter(a1, a2, b1, b2):
        x1, y1 = a1
        x2, y2 = a2
        x3, y3 = b1
        x4, y4 = b2
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    V = inter(A1, A2, B1, B2)
    s += arc(V[0], V[1], 56, A1, B1)
    s += dot(*A1, 4)
    s += dot(*B1, 4)
    s += txt(V[0] - 66, V[1] - 4, "≈1–2°", "v", "end")
    s += txt(104, 162, "femoral joint line", "lA")
    s += txt(104, 232, "tibial joint line", "lB")
    s += foot(W, H, "Convergence of femoral and tibial joint lines (exaggerated).")
    return "jlca.svg", s


# ---------------------------------------------------------------------------
# 11. MAD
# ---------------------------------------------------------------------------
def mad():
    W, H = 300, 520
    s = hdr(W, H, "Mechanical-axis deviation (MAD)",
            "Coronal — knee centre to Mikulicz line")
    Hh = (120, 92)
    Ank = (154, 486)
    K = (150, 296)
    s += poly([(106, 100), (142, 108), (168, 292), (140, 296)], "bone2")
    s += poly([(142, 302), (168, 298), (158, 486), (134, 486)], "bone2")
    s += circle(116, 96, 20, "bone2")
    s += line(Hh[0], Hh[1], Ank[0], Ank[1], "A")       # Mikulicz line
    F = foot_perp(K, Hh, Ank)
    s += darrow(F[0], F[1], K[0], K[1])                # MAD
    s += dot(*Hh)
    s += dot(*Ank)
    s += dot(*K)
    s += dot(*F, 4)
    s += txt(Hh[0] + 8, Hh[1] - 6, "hip", "lA")
    s += txt(Ank[0] + 8, Ank[1] + 6, "ankle", "lA")
    s += txt(K[0] + 12, K[1] + 4, "knee", "l")
    s += txt(K[0] + 14, K[1] + 22, "≈8 mm medial", "lC")
    s += foot(W, H, "Knee centre to the Mikulicz (hip→ankle) line.")
    return "mad.svg", s


# ---------------------------------------------------------------------------
# 12. Subchondral distance (hip joint space)
# ---------------------------------------------------------------------------
def subchondral_distance():
    """Ray-tracing schematic: a cone of rays cast from the femoral head centre.

    Mirrors ``measurements/hip.py::_subchondral_distance_ray_tracing``: a fan of
    rays leaves the femoral head centre (FHC) inside a cone aimed at the
    acetabular centroid; per ray the femoral-head-contour exit and the first
    acetabulum hit are found and their physical distance recorded; the reported
    value is the mean over all rays. Drawing geometry is computed from a common
    centre so every ray's exit sits exactly on the head contour.
    """
    W, H = 400, 452
    s = hdr(W, H, "Subchondral distance (hip joint space)",
            "Rays from the femoral head centre through the joint space")
    C = (196, 330)          # femoral head centre (FHC)
    R = 76                  # femoral head radius
    Ri, Ro = 122, 154       # acetabular inner / outer surface radius from C
    th0 = math.radians(-96)  # cone axis: up, slightly medial, toward the acetabulum
    cone = math.radians(45)  # cone half-angle (matches the default cone_angle=45°)

    def P(rad, th):
        return (C[0] + rad * math.cos(th), C[1] + rad * math.sin(th))

    def ray(x1, y1, x2, y2, w=1.4, op=0.75):  # thin traced ray (red, like the 3D plot)
        return (f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="#DC2626" stroke-width="{w}" stroke-opacity="{op}" '
                f'stroke-linecap="round"/>\n')

    # femoral shaft descending from the head, then the head itself
    s += neck_poly((C[0] - 2, C[1] + 34), (C[0] + 30, H - 6), 30)
    s += circle(*C, R)

    # acetabular roof: a curved band above the head (sampled arcs -> robust fill)
    span = math.radians(58)
    a1, a2 = th0 - span, th0 + span
    n = 26
    outer = [P(Ro, a1 + (a2 - a1) * k / n) for k in range(n + 1)]
    inner = [P(Ri, a2 - (a2 - a1) * k / n) for k in range(n + 1)]
    s += poly(outer + inner, "bone2")

    # cone boundaries (dashed grey) + the fan of traced rays (FHC -> acetabulum)
    s += line(*C, *P(Ro + 16, th0 - cone), "C")
    s += line(*C, *P(Ro + 16, th0 + cone), "C")
    for off in (-45, -30, -15, 15, 30, 45):
        h = P(Ri, th0 + math.radians(off))
        s += ray(C[0], C[1], h[0], h[1])

    # cone axis -> acetabular centroid, and the cone half-angle arc
    cen = P((Ri + Ro) / 2, th0)
    s += line(*C, *cen, "A")
    s += arc(C[0], C[1], 60, cen, P(60, th0 + cone))
    s += dot(*cen, 4)

    # one highlighted ray: femoral-head exit -> first acetabulum hit = the distance
    thm = th0 + math.radians(30)
    ex, ht = P(R, thm), P(Ri, thm)
    s += ray(C[0], C[1], ex[0], ex[1], w=2.4, op=1.0)   # emphasise only the head-interior part
    s += line(ex[0], ex[1], ht[0], ht[1], "dist")       # purple gap = the measured distance
    s += dot(*ex)
    s += dot(*ht)
    s += dot(*C, 4)

    # labels
    s += txt(C[0] - 12, C[1] + 5, "FHC", "l", "end")
    s += txt(182, 150, "acetabulum", "l", "middle")
    s += txt(70, 372, "femoral head", "l")
    s += txt(cen[0] - 6, cen[1] - 8, "cone axis", "lA", "end")
    s += txt(232, 292, "45°", "v")
    s += txt(ex[0] + 10, ex[1] + 14, "head exit", "l")
    s += txt(ht[0] + 8, ht[1] - 8, "acetabulum hit", "l")
    s += txt(256, 244, "≈4 mm", "lC")
    s += foot(W, H, "Mean femoral-head→acetabulum gap over a 45° cone of rays.")
    return "subchondral_distance.svg", s


DIAGRAMS = [
    femoral_torsion, tibial_torsion, knee_rotation, ccd_angle, bone_length,
    acetabular_version, ce_angle, femoral_offset, hka_angle, jlca, mad,
    subchondral_distance,
]


def main():
    for fn in DIAGRAMS:
        name, svg = fn()
        with open(os.path.join(OUT, name), "w") as f:
            f.write(svg)
        print("wrote", name)


if __name__ == "__main__":
    main()
